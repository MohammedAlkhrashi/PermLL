import argparse
import wandb
from torch.nn.modules import CrossEntropyLoss
from callbacks import (
    CallbackLearningRateScheduler,
    CallbackNoisyStatistics,
    CallbackPermutationStats,
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


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--networks_lr", type=float, default=0.01)
    parser.add_argument("--permutation_lr", type=float, default=10.0)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pretrained", type=str2bool, default=False)
    parser.add_argument("--disable_perm", type=str2bool, default=True)
    parser.add_argument("--with_lr_scheduler", type=str2bool, default=True)
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--networks_optim", type=str, default="adam")
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--upperbound_exp", type=str2bool, default=False)
    parser.add_argument("--networks_per_group", type=int, default=1)
    parser.add_argument("--num_groups", type=int, default=1)
    parser.add_argument("--change_every", type=int, default=1)
    parser.add_argument("--gpu_num", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--generations", type=int, default=1)
    args = parser.parse_args()
    return vars(args)


def main():
    config = get_config()
    wandb.init(project="test-project", entity="nnlp", config=config)
    train_transform = create_train_transform()
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
    perm_model = None
    for _ in range(config["generations"]):
        model: GroupModel = create_group_model(
            config["networks_per_group"] * config["num_groups"],
            num_classes=10,
            pretrained=config["pretrained"],
            dataset_targets=loaders["train"].dataset.noisy_labels,
            disable_perm=config["disable_perm"],
        )

        print(f"alpha matirx before: {model.perm_model.alpha_matrix.sum()}")
        if perm_model:
            model.perm_model = perm_model
            print(f"alpha matirx after: {model.perm_model.alpha_matrix.sum()}")

        optimizer = create_group_optimizer(
            model,
            networks_optim_choice=config["networks_optim"],
            networks_lr=config["networks_lr"],
            permutation_lr=config["permutation_lr"],
            weight_decay=config["weight_decay"],
        )
        callbacks = [CallbackNoisyStatistics(), CallbackPermutationStats()]
        if config["with_lr_scheduler"]:
            callbacks.append(
                CallbackLearningRateScheduler(
                    optimizer.network_optimizer,
                    config["networks_lr"],
                    config["epochs"],
                    steps_per_epoch=len(loaders["train"]),
                )
            )
        TrainPermutation(
            model=model,
            optimizer=optimizer,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            epochs=config["epochs"],
            criterion=CrossEntropyLoss(),
            gpu_num=config["gpu_num"],
            group_picker=group_picker,
            callbacks=callbacks,
            grad_clip=config["grad_clip"],
        ).start()
        perm_model = model.perm_model


if __name__ == "__main__":
    main()
