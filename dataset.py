import random
from collections import Counter

import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from autoaugment import CIFAR10Policy


class NoisyDataset(Dataset):
    def __init__(self, images, clean_labels, noisy_labels, transforms=None) -> None:
        self.images = images
        self.original_labels = clean_labels
        self.noisy_labels = noisy_labels
        self.transforms = transforms
        print(transforms)

    def __getitem__(self, index):
        item = dict()
        item["image"] = self.images[index]
        if self.transforms:
            item["image"] = self.transforms(item["image"])
        item["true_label"] = self.original_labels[index]
        item["noisy_label"] = self.noisy_labels[index]
        item["sample_index"] = index
        return item

    def __len__(self):
        return len(self.noisy_labels)


class CIFAR10N(Dataset):
    def __init__(self, dataset: Dataset, noise_mode) -> None:
        noise_file = torch.load("./CIFAR-N/CIFAR-10_human.pt")
        original_labels = torch.tensor(dataset.targets)
        noisy_labels = noise_file[noise_mode]
        # ["worse_label"]
        # ["aggre_label"]
        # ["random_label1"]
        # ["random_label2"]
        # ["random_label3"]

    def __getitem__(self, index):
        item = dict()
        item["image"], item["true_label"] = self.dataset[index]
        item["noisy_label"] = self.noisy_labels[index]
        item["sample_index"] = index

        return item

    def __len__(self):
        return len(self.dataset)
