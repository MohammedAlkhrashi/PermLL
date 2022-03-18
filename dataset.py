import random
from collections import Counter

import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


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


class NoisyDataset(Dataset):
    def __init__(self, dataset: Dataset, noise=0.3, upperbound=False) -> None:
        self.dataset = dataset
        self.original_labels = torch.tensor(dataset.targets)
        self.noisy_labels = apply_noise(self.original_labels, noise)
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


def cifar_10_dataloaders(batch_size, noise, num_workers, upperbound=False):

    data_folder = "./dataset"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform
    )

    valset = torchvision.datasets.CIFAR10(
        root=data_folder, train=False, download=True, transform=transform
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
