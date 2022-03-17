import wandb
from torch.nn.modules import CrossEntropyLoss
from callbacks import CallbackNoisyStatistics

from dataset import cifar_10_dataloaders
from group_utils import GroupPicker, create_group_model, create_group_optimizer
from model import GroupModel
from train import TrainPermutation


def get_config():
    config = dict()
    # Training
    config["networks_lr"] = 0.05
    config["permutation_lr"] = 0.01
    config["epochs"] = 100
    config["batch_size"] = 16
    config["pretrained"] = False
    # Noise related
    config["noise"] = 0
    config["upperbound_exp"] = False
    # Group related
    config["networks_per_group"] = 5
    config["num_groups"] = 3
    config["change_every"] = 3
    # General config
    config["gpu_num"] = "0"
    config["num_workers"] = 8
    return config


def main():
    config = get_config()
    wandb.init(project="test-project", entity="nnlp", config=config)
    loaders = cifar_10_dataloaders(
        batch_size=config["batch_size"],
        noise=config["noise"],
        num_workers=config["num_workers"],
        upperbound=config["upperbound_exp"],
    )
    model: GroupModel = create_group_model(
        config["networks_per_group"] * config["num_groups"],
        num_classes=10,
        pretrained=config["pretrained"],
        dataset_targets=loaders["train"].dataset.noisy_labels,
    )
    optimizer = create_group_optimizer(
        model,
        networks_lr=config["networks_lr"],
        permutation_lr=config["permutation_lr"],
    )
    callbacks = [CallbackNoisyStatistics()]
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
    ).start()


if __name__ == "__main__":
    main()

