import argparse

import torch
from torch.nn.modules import CrossEntropyLoss

import wandb
from callbacks import (
    CallbackGroupPickerReseter,
    CallbackLearningRateScheduler,
    CallbackNoisyStatistics,
    CallbackPermutationStats,
    CallbackLabelCorrectionStats,
)
from dataset import cifar_10_dataloaders, create_train_transform
from group_utils import GroupPicker, create_group_model, create_group_optimizer
from model import GroupModel
from train import TrainPermutation


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
    parser.add_argument("--with_lr_scheduler", type=str2bool, default=True)
    parser.add_argument("--grad_clip", type=float, default=-1)
    parser.add_argument("--networks_optim", type=str, default="adam")
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument(
        "--augmentation",
        type=str,
        default="default",
        choices=["AutoAugment", "default"],
    )
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument(
        "--upperbound_exp", type=str2bool, default=False
    )  # do we need this hparam?
    parser.add_argument("--networks_per_group", type=int, default=1)
    parser.add_argument("--num_groups", type=int, default=1)
    parser.add_argument("--change_every", type=int, default=1)
    parser.add_argument("--gpu_num", type=str, default="0")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--num_workers", type=int, default=15)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--perm_init_value", type=int, default=4)
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
    args = parser.parse_args()
    return vars(args)


def main():
    config = get_config()
    wandb.init(project="test-project", entity="nnlp", config=config)
    train_transform = create_train_transform(config["augmentation"])
    loaders = cifar_10_dataloaders(
        batch_size=config["batch_size"],
        noise=config["noise"],
        num_workers=config["num_workers"],
        upperbound=config["upperbound_exp"],
        train_transform=train_transform,
    )
    group_picker = GroupPicker(
        networks_per_group=config["networks_per_group"],
        num_groups=config["num_groups"],
        change_every=config["change_every"]
        if config["change_every"] != -1
        else len(loaders["train"]),
    )

    for gen in range(config["num_generations"]):
        model: GroupModel = create_group_model(
            config["networks_per_group"] * config["num_groups"],
            num_classes=10,
            pretrained=config["pretrained"],
            dataset_targets=loaders["train"].dataset.noisy_labels,
            perm_init_value=config["perm_init_value"],
            model_name=config["model_name"],
            disable_perm=config["disable_perm"],
        )

        optimizer = create_group_optimizer(
            model,
            networks_optim_choice=config["networks_optim"],
            networks_lr=config["networks_lr"],
            permutation_lr=config["permutation_lr"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
        )
        callbacks = [
            CallbackNoisyStatistics(),
            CallbackPermutationStats(),
            CallbackLabelCorrectionStats(),
        ]
        if config["with_lr_scheduler"]:
            callbacks.append(
                CallbackLearningRateScheduler(
                    optimizer.network_optimizer,
                    config["networks_lr"],
                    config["epochs"],
                    steps_per_epoch=len(loaders["train"]),
                )
            )
        if config["reshuffle_groups"]:
            callbacks.append(CallbackGroupPickerReseter(group_picker))

        TrainPermutation(
            model=model,
            optimizer=optimizer,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            epochs=config["epochs"],
            criterion=CrossEntropyLoss(label_smoothing=config["label_smoothing"]),
            gpu_num=config["gpu_num"],
            group_picker=group_picker,
            callbacks=callbacks,
            grad_clip=config["grad_clip"],
            num_permutation_limit=config["num_permutation_limit"],
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
