import argparse

import torch
from torch.nn.modules import CrossEntropyLoss, NLLLoss, MSELoss, L1Loss, KLDivLoss

import wandb
from callbacks import (
    AdaptivePermLRScheduler,
    CallbackGroupPickerReseter,
    CallbackLabelCorrectionStats,
    CallbackNoisyStatistics,
    CallbackPermutationStats,
    CosineAnnealingLRScheduler,
    IdentityCallback,
    OneCycleLearningRateScheduler,
    StepLRLearningRateScheduler,
    AdaptiveNetworkLRScheduler,
)
from data_utils import create_dataloaders, create_train_transform
from group_utils import (
    GroupPicker,
    NLLSmoothing,
    create_group_model,
    create_group_optimizer,
)
from model import GroupModel
from train import TrainPermutation

# torch.autograd.set_detect_anomaly(True)


def str2list(v):
    """
    example: v="100,150" return [100,150]
    """
    return list(map(int, v.split(",")))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_loss_func(loss_func, logits_softmax_mode, config, reduction="none"):
    log_space_prediction = "log" in logits_softmax_mode
    prop_space_prediction = "softmax" == logits_softmax_mode
    if loss_func == "mse":
        print("Using MSE loss function")
        assert prop_space_prediction
        return MSELoss(reduction=reduction)
    elif loss_func == "mae":
        print("Using MAE loss function")
        assert prop_space_prediction
        return L1Loss(reduction=reduction)
    elif loss_func == "kl":
        print("Using KL loss function")
        assert log_space_prediction
        return KLDivLoss(reduction="batchmean")
    elif loss_func == "ce":
        print("Using CE loss function")
        if log_space_prediction:
            print("Using softmax before permutation, (NLLLoss Criterion)")
            return NLLSmoothing(
                smoothing=config["label_smoothing"], reduction=reduction
            )
        else:
            return CrossEntropyLoss(
                label_smoothing=config["label_smoothing"], reduction=reduction
            )


def create_adaptive_perm_lr(
    optimizer, perm_lr, threshold, adaptive_lr_mode, disable=False
):
    if disable:
        print("Using identity learning rate scheduler for perm layer")
        return IdentityCallback()
    return AdaptivePermLRScheduler(optimizer, threshold, adaptive_lr_mode, perm_lr)


def create_lr_scheduler(lr_scheduler, optimizer, loaders, config):
    if lr_scheduler == "one_cycle":
        return OneCycleLearningRateScheduler(
            optimizer,
            max_lr=config["networks_lr"],
            epochs=config["epochs"],
            steps_per_epoch=len(loaders["train"]),
            final_div_factor=config["final_div_factor"],
        )
    elif lr_scheduler == "cosine":
        return CosineAnnealingLRScheduler(
            optimizer,
            T_0=config["t_zero"]
            * len(loaders["train"], T_mult=config["t_mult"] * len(loaders["train"])),
        )
    elif lr_scheduler == "step_lr":
        return StepLRLearningRateScheduler(
            optimizer,
            milestones=config["milestones"],
            gamma=config["gamma"],
        )
    elif lr_scheduler == "adaptive":
        return AdaptiveNetworkLRScheduler(
            optimizer,
            config["networks_lr"],
            config["batch_size"] // 5,
            0.8,
            factor=100000,
        )

    elif lr_scheduler == "default":
        print("Using identity learning rate scheduler for networks")
        return IdentityCallback()
    else:
        raise NotImplementedError


def get_new_labels(
    new_label_source, prediction_before_perm, sample_index, alpha_matrix
):
    if new_label_source == "alpha_matrix":
        _, alpha_label = torch.max(alpha_matrix.detach().cpu(), 1)
        return alpha_label
    elif new_label_source == "prediction_before_perm":
        sample_pred = zip(sample_index.tolist(), prediction_before_perm.tolist())
        sample_pred_sorted = sorted(sample_pred, key=lambda x: x[0])
        pred_sorted = torch.tensor(list(zip(*sample_pred_sorted))[1])
        return pred_sorted


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--networks_lr", type=float, default=0.05)
    parser.add_argument("--permutation_lr", type=float, default=80.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrained", type=str2bool, default=False)
    parser.add_argument("--disable_perm", type=str2bool, default=False)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="default",
        choices=["default", "one_cycle", "cosine", "adaptive"],
    )
    parser.add_argument("--grad_clip", type=float, default=-1)
    parser.add_argument("--networks_optim", type=str, default="adam")
    parser.add_argument("--perm_optim", type=str, default="sgd")
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument(
        "--augmentation",
        type=str,
        default="default",
        choices=["AutoAugment", "default"],
    )
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--noise_mode", type=str, default="sym")
    parser.add_argument("--networks_per_group", type=int, default=1)
    parser.add_argument("--num_groups", type=int, default=1)
    parser.add_argument("--change_every", type=int, default=1)
    parser.add_argument("--gpu_num", type=str, default="0")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--num_workers", type=int, default=15)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--init_max_prob", type=float, default=0.7)
    parser.add_argument(
        "--num_permutation_limit",
        type=int,
        default=-1,
        help="maximum number of permutation allowed per generation, -1 means no limit",
    )
    parser.add_argument(
        "--new_label_source",
        type=str,
        default="alpha_matrix",
        choices=["alpha_matrix", "prediction_before_perm"],
    )
    parser.add_argument("--reshuffle_groups", type=str2bool, default=False)
    parser.add_argument("--avg_before_perm", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--milestones", type=str2list, default="150")
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=20,
        help="maximum number of iteration with no improvement, use -1 for no early stopping",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "cloth"],
    )
    parser.add_argument("--perm_momentum", type=float, default=0)
    parser.add_argument("--with_adaptive_perm_lr", type=str2bool, default=False)
    parser.add_argument("--adaptive_min_acc", type=float, default=0.1)
    parser.add_argument(
        "--adaptive_lr_mode", type=str, default="linear", choices=["linear", "constant"]
    )
    parser.add_argument("--softmax_temp", type=float, default=1)
    parser.add_argument("--final_div_factor", type=float, default=1e4)
    parser.add_argument("--t_zero", type=int, default=50)
    parser.add_argument("--t_mult", type=int, default=1)
    parser.add_argument("--equalize_losses", type=str2bool, default=False)
    parser.add_argument(
        "--logits_softmax_mode",
        type=str,
        default="default",
        choices=["default", "log_softmax", "softmax", "log_perm_softmax"],
    )
    parser.add_argument(
        "--loss_func",
        type=str,
        default="ce",
        choices=["ce", "mse", "mae", "kl"],
    )

    args = parser.parse_args()
    print(args)
    return vars(args)


def main():
    config = get_config()
    wandb.init(project="test-project", entity="nnlp", config=config)
    train_transform = create_train_transform(config["augmentation"])
    loaders, num_classes = create_dataloaders(
        dataset_name=config["dataset"],
        noise=config["noise"],
        noise_mode=config["noise_mode"],
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        train_transform=train_transform,
    )
    group_picker = GroupPicker(
        networks_per_group=config["networks_per_group"],
        num_groups=config["num_groups"],
        change_every=config["change_every"]
        if config["change_every"] != -1
        else len(loaders["train"]),
    )

    for _ in range(config["num_generations"]):
        model: GroupModel = create_group_model(
            config["networks_per_group"] * config["num_groups"],
            num_classes=num_classes,
            pretrained=config["pretrained"],
            dataset_targets=loaders["train"].dataset.noisy_labels,
            init_max_prob=config["init_max_prob"],
            model_name=config["model_name"],
            avg_before_perm=config["avg_before_perm"],
            disable_perm=config["disable_perm"],
            softmax_temp=config["softmax_temp"],
            logits_softmax_mode=config["logits_softmax_mode"],
        )

        optimizer = create_group_optimizer(
            model,
            networks_optim_choice=config["networks_optim"],
            networks_lr=config["networks_lr"],
            permutation_lr=0
            if config["with_adaptive_perm_lr"]
            else config["permutation_lr"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
            perm_optimizer=config["perm_optim"],
            perm_momentum=config["perm_momentum"],
        )
        callbacks = [
            CallbackNoisyStatistics(max_no_improvement=config["early_stopping"]),
            CallbackPermutationStats(),
            CallbackLabelCorrectionStats(),
            create_lr_scheduler(
                config["lr_scheduler"], optimizer.network_optimizer, loaders, config
            ),
            create_adaptive_perm_lr(
                optimizer.permutation_optimizer,
                perm_lr=config["permutation_lr"],
                threshold=config["adaptive_min_acc"],
                adaptive_lr_mode=config["adaptive_lr_mode"],
                disable=not config["with_adaptive_perm_lr"],
            ),
        ]
        if config["reshuffle_groups"]:
            callbacks.append(CallbackGroupPickerReseter(group_picker))

        criterion: torch.nn.Module = create_loss_func(
            config["loss_func"], config["logits_softmax_mode"], config
        )

        TrainPermutation(
            model=model,
            optimizer=optimizer,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            epochs=config["epochs"],
            criterion=criterion,
            gpu_num=config["gpu_num"],
            group_picker=group_picker,
            callbacks=callbacks,
            grad_clip=config["grad_clip"],
            num_permutation_limit=config["num_permutation_limit"],
            equalize_losses=config["equalize_losses"],
        ).start()

        new_label = get_new_labels(
            config["new_label_source"],
            callbacks[0].all_train_prediction_before_perm,
            callbacks[0].all_sample_index,
            model.perm_model.alpha_matrix,
        )  # TODO: fix callbacks[0]
        loaders["train"].dataset.noisy_labels = new_label


if __name__ == "__main__":
    main()
