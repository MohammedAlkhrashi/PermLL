import random
from collections import Counter

import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from autoaugment import CIFAR10Policy

NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def apply_noise(labels: Tensor, noise: float):
    random.seed(42)

    labels = labels.clone()
    classes = labels.unique()

    total_labels_count = len(labels)
    labels_to_change_count = int(total_labels_count * noise)

    idxs_to_change = random.sample(range(total_labels_count), k=labels_to_change_count)
    for idx in idxs_to_change:
        cur_label = labels[idx]
        rand_class_idx = random.randint(0, len(classes) - 2)
        new_label = classes[classes != cur_label][rand_class_idx]
        labels[idx] = new_label
    return labels


def balanced_apply_noise(labels: Tensor, noise: float):
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


class NoisyDataset(Dataset):
    def __init__(self, dataset: Dataset, noise=0.3, upperbound=False) -> None:
        self.dataset = dataset
        self.original_labels = torch.tensor(dataset.targets)
        self.noisy_labels = balanced_apply_noise(self.original_labels, noise)
        self.same_indices = torch.eq(self.original_labels, self.noisy_labels)

        if upperbound:
            self.original_labels = self.original_labels[self.same_indices]
            self.noisy_labels = self.noisy_labels[self.same_indices]
            self.dataset.data = self.dataset.data[self.same_indices]
            self.dataset.targets = list(
                torch.tensor(self.dataset.targets)[self.same_indices]
            )
        if noise == 0:
            print("Noise set to 0")
            assert torch.equal(self.original_labels, self.noisy_labels)

    def __getitem__(self, index):
        item = dict()
        item["image"], item["true_label"] = self.dataset[index]
        item["noisy_label"] = self.noisy_labels[index]
        item["sample_index"] = index

        return item

    def __len__(self):
        return len(self.dataset)


def cifar_dataloaders(
    batch_size, noise, num_workers, train_transform, upperbound=False, cifar100=False
):
    data_folder = "./dataset"
    dataset = (
        torchvision.datasets.CIFAR100 if cifar100 else torchvision.datasets.CIFAR10
    )
    val_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*NORMALIZATION)]
    )
    trainset = dataset(
        root=data_folder, train=True, download=True, transform=train_transform
    )

    valset = dataset(
        root=data_folder, train=False, download=True, transform=val_transform
    )

    def print_label_distribution(dataset):
        labels = []
        for d, l in zip(dataset.data, dataset.targets):
            labels.append(l)

        print("Class dist:")
        print(Counter(labels))

    print_label_distribution(trainset)
    print_label_distribution(valset)

    trainset = NoisyDataset(trainset, noise=noise, upperbound=upperbound)
    valset = NoisyDataset(valset, noise)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valloader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return {
        "train": trainloader,
        "val": valloader,
    }


def create_train_transform(augmentation):
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
    train_transform = transforms.Compose(transforms_list)
    return train_transform
