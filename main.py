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


def get_config():
    config = dict()
    # Training
    config["networks_lr"] = 0.01
    config["permutation_lr"] = 100 / (10 * 1)
    config["weight_decay"] = 1e-4
    config["epochs"] = 200
    config["batch_size"] = 32
    config["pretrained"] = False
    config["disable_perm"] = True
    config["with_lr_scheduler"] = True
    config["grad_clip"] = 0.1  # -1 to turn off
    config["networks_optim"] = "adam"  # sgd,adam
    # Noise relate
    config["noise"] = 0.3
    config["upperbound_exp"] = False
    # Group related
    config["networks_per_group"] = 1
    config["num_groups"] = 1
    config["change_every"] = 1
    # General config
    config["gpu_num"] = "0"
    config["num_workers"] = 4
    return config


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
    model: GroupModel = create_group_model(
        config["networks_per_group"] * config["num_groups"],
        num_classes=10,
        pretrained=config["pretrained"],
        dataset_targets=loaders["train"].dataset.noisy_labels,
        disable_perm=config["disable_perm"],
    )
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
    group_picker = GroupPicker(
        networks_per_group=config["networks_per_group"],
        num_groups=config["num_groups"],
        change_every=config["change_every"]
        if config["change_every"] != -1
        else len(loaders["train"]),
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


if __name__ == "__main__":
    main()
