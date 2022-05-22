import random

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from autoaugment import CIFAR10Policy
from dataset import NoisyDataset

NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def apply_sym_noise(labels: Tensor, noise: float):
    random.seed(42)
    original = labels.clone()
    labels = labels.clone()
    classes = labels.unique()
    labels_per_class = int(len(labels) / len(classes))
    labels_to_change_per_class = int(labels_per_class * noise)
    for cur_class in classes:
        other_classes = classes[classes != cur_class]
        idxs_to_change = random.sample(
            range(labels_per_class), k=labels_to_change_per_class
        )
        idxs_with_cur_class_label = torch.where(original == cur_class)[0]
        counter = cur_class.item() - 1  # random.randint(0, len(other_classes)-1)
        for idx in idxs_to_change:
            labels[idxs_with_cur_class_label[idx]] = other_classes[counter]
            counter = (counter + 1) % len(other_classes)
    return labels


def apply_noise(labels, noise, noise_mode):
    if noise_mode == "sym":
        return apply_sym_noise(labels, noise)
    elif noise_mode == "asym":
        raise NotImplementedError
    else:
        cifar_10_human_noise = torch.load("./CIFAR-N/CIFAR-10_human.pt")
        cifar_100_human_noise = torch.load("./CIFAR-N/CIFAR-100_human.pt")
        if noise_mode in cifar_10_human_noise:
            return torch.tensor(cifar_10_human_noise[noise_mode])
        if noise_mode in cifar_100_human_noise:
            return torch.tensor(cifar_100_human_noise[noise_mode])


def prepare_dataset(dataset_name, noise,noise_mode):

    dataset_items = dict()
    dataset_items["train"] = dict()
    dataset_items["val"] = dict()
    dataset_items["test"] = dict()
    if dataset_name == "cifar10":
        data_folder = "./dataset"
        trainset = torchvision.datasets.CIFAR10(
            root=data_folder, train=True, download=True
        )
        valset = torchvision.datasets.CIFAR10(
            root=data_folder, train=False, download=True
        )
    if dataset_name == "cifar100":
        data_folder = "./dataset"
        trainset = torchvision.datasets.CIFAR100(
            root=data_folder, train=True, download=True
        )
        valset = torchvision.datasets.CIFAR100(
            root=data_folder, train=False, download=True
        )
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        dataset_items["train"]["images"] = trainset.data
        dataset_items["train"]["clean_labels"] = torch.tensor(trainset.targets)
        dataset_items["train"]["noisy_labels"] = apply_noise(
            torch.tensor(trainset.targets), noise=noise, noise_mode=noise_mode
        )

        dataset_items["val"]["images"] = valset.data
        dataset_items["val"]["clean_labels"] = torch.tensor(valset.targets)
        dataset_items["val"]["noisy_labels"] = apply_noise(
            torch.tensor(valset.targets), noise=noise, noise_mode="sym"
        )
        dataset_items["val"]["transforms"] = transforms.Compose(
            [
                Image.fromarray,
                transforms.ToTensor(),
                transforms.Normalize(*NORMALIZATION),
            ]
        )

    elif dataset_name == "cloth":
        pass
    else:
        raise ValueError()

    return dataset_items


def create_dataloaders(dataset_name, batch_size, num_workers, noise, train_transform,noise_mode):
    dataset_items = prepare_dataset(dataset_name, noise,noise_mode)
    dataset_items["train"][
        "transforms"
    ] = train_transform  # TODO: move to prepare_dataset
    train_set = NoisyDataset(**dataset_items["train"])
    val_set = NoisyDataset(**dataset_items["val"])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return {
        "train": train_loader,
        "val": val_loader,
    }


def create_train_transform(augmentation, dataset="cifar10"):
    transforms_list = []
    if augmentation == "default":
        transforms_list.append(
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        )
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(*NORMALIZATION))
    elif augmentation == "AutoAugment":
        transforms_list.append(transforms.RandomCrop(32, padding=4, fill=128))
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(CIFAR10Policy())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(*NORMALIZATION))
    else:
        raise NotImplementedError

    if "cifar10" in dataset:
        transforms_list.insert(0, Image.fromarray)
    train_transform = transforms.Compose(transforms_list)

    return train_transform