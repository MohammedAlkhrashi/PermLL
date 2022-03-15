import wandb
from torch.nn.modules import CrossEntropyLoss
from torch.optim import SGD

from dataset import cifar_10_dataloaders
from model import GroupModel, create_group_model
from train import TrainPermutation


def get_config():
    config = dict()
    config["learning_rate"] = 0.001
    config["epochs"] = 100
    config["batch_size"] = 16
    config["noise"] = 0
    config["gpu_num"] = "0"
    config["num_workers"] = 8
    config["upperbound_exp"] = False
    config["num_networks"] = 3
    config["pretrained"] = False
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
        config["num_networks"], num_classes=10, pretrained=config["pretrained"], dataset_targets=loaders['train'].dataset.targets
    )
    optimizer = SGD(model.parameters(), lr=config["learning_rate"])
    TrainPermutation(
        model=model,
        optimizer=optimizer,
        train_loader=loaders["val"],
        val_loader=loaders["val"],
        epochs=config["epochs"],
        criterion=CrossEntropyLoss(),
        gpu_num=config["gpu_num"],
    ).start()


if __name__ == "__main__":
    main()

