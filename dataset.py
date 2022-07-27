import random
from collections import Counter
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from autoaugment import CIFAR10Policy
from sklearn.metrics import confusion_matrix


def print_transation_matrix(clean_labels, noisy_labels):
    cm = confusion_matrix(clean_labels.numpy(), noisy_labels.numpy())
    classes = clean_labels.unique()
    if len(classes) == 0:
        return
    samples_per_class = int(len(clean_labels) / len(classes))
    print(cm / samples_per_class)


class NoisyDataset(Dataset):
    def __init__(self, images, clean_labels, noisy_labels, transforms=None) -> None:
        self.images = images
        self.original_labels = clean_labels
        self.noisy_labels = noisy_labels
        self.transforms = transforms

        print(transforms)
        print_transation_matrix(clean_labels, noisy_labels)

    def __getitem__(self, index):
        item = dict()
        item["image"] = self.images[index]
        if isinstance(item["image"], str):
            item["image"] = Image.open(item["image"]).convert("RGB")
        if self.transforms:
            item["image"] = self.transforms(item["image"])
        item["true_label"] = self.original_labels[index]
        item["noisy_label"] = self.noisy_labels[index]
        item["sample_index"] = index
        return item

    def __len__(self):
        return len(self.noisy_labels)
