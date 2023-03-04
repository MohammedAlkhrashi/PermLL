import random

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from autoaugment import CIFAR10Policy
from dataset import NoisyDataset
import numpy as np
from sklearn.model_selection import train_test_split


NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def apply_sym_noise(labels: Tensor, noise: float):
    random.seed(42)
    original = labels.clone()
    labels = labels.clone()
    classes = labels.unique()
    labels_per_class = int(len(labels) / len(classes))
    labels_to_change_per_class = int(labels_per_class * noise)
    for cur_class in classes:
        idxs_to_change = random.sample(
            range(labels_per_class), k=labels_to_change_per_class
        )
        idxs_with_cur_class_label = torch.where(original == cur_class)[0]
        counter = cur_class.item() - 1  # random.randint(0, len(other_classes)-1)
        for idx in idxs_to_change:
            labels[idxs_with_cur_class_label[idx]] = classes[counter]
            counter = (counter + 1) % len(classes)
    return labels


def apply_asym_noise(labels: Tensor, noise: float, num_classes, corruption_map=None):
    if corruption_map is None:
        corruption_map = {i: (i + 1) % num_classes for i in range(num_classes)}

    original = labels.clone()
    labels = labels.clone()
    classes = labels.unique()
    labels_per_class = int(len(labels) / len(classes))
    labels_to_change_per_class = int(labels_per_class * noise)
    for cur_class in classes:
        idxs_to_change = random.sample(
            range(labels_per_class), k=labels_to_change_per_class
        )
        idxs_with_cur_class_label = torch.where(original == cur_class)[0]
        labels[idxs_with_cur_class_label[idxs_to_change]] = corruption_map.get(
            int(cur_class), int(cur_class)
        )

    return labels


def get_asym_corruption_map(dataset_name, noise_mode):
    if dataset_name == "cifar10":
        corruption_map = {
            9: 1,  # truck -> automobile
            2: 0,  # bird -> airplane
            3: 5,  # cat -> dog
            5: 3,  # dog -> cat
            4: 7,  # deer -> horse
        }
        if noise_mode == "asym2":
            del corruption_map[5]  # some methods remove dog -> cat
        return corruption_map
    if dataset_name == "cifar100":
        # change subclasses (5 per superclass) within the same superclass (20 total superclasses)
        # (e.g 5->6, 6>7, 7>8, 8>9, 9>5)

        corruption_map = {}
        for super in range(20):
            for sub in range(5):
                from_sub = (super * 5) + sub
                to_sub = (super * 5) + (sub + 1) % 5
                corruption_map[from_sub] = to_sub
        return corruption_map


def apply_noise(labels, noise, noise_mode, dataset_name):
    if len(labels) == 0:
        return labels
    if noise_mode == "sym":
        return apply_sym_noise(labels, noise)
    elif noise_mode == "asym" or noise_mode == "asym2":
        num_classes = 10 if dataset_name == "cifar10" else 100
        corruption_map = get_asym_corruption_map(dataset_name, noise_mode)

        return apply_asym_noise(
            labels, noise, num_classes=num_classes, corruption_map=corruption_map
        )
    elif noise_mode == "custom":
        custom_labels = torch.load("./diff_indices_test/all_noisy_labels.pt")
        custom_indices = torch.load("./diff_indices_test/difficult_indices.pt")
        labels[custom_indices] = custom_labels[custom_indices]
        return labels
    else:
        cifar_10_human_noise = torch.load("./CIFAR-N/CIFAR-10_human.pt")
        cifar_100_human_noise = torch.load("./CIFAR-N/CIFAR-100_human.pt")
        if noise_mode in cifar_10_human_noise:
            return torch.tensor(cifar_10_human_noise[noise_mode])
        if noise_mode in cifar_100_human_noise:
            return torch.tensor(cifar_100_human_noise[noise_mode])


def txt_to_list(path):
    with open(path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def txt_to_dict(path):
    with open(path) as file:
        lines = file.readlines()
        lines = [line.rstrip().split() for line in lines]
    return dict(lines)


def get_cloth1m_paths_labels(map_path, keys_path, root="./Cloth1M/"):
    map_path_noisy_label = txt_to_dict(f"{root}/annotations/{map_path}")
    paths = txt_to_list(f"{root}/annotations/{keys_path}")
    labels = [int(map_path_noisy_label[path]) for path in paths]
    paths = [f"{root}/{path}" for path in paths]
    return paths, labels


def get_webvision_paths_labels(info_file, root, val=False):
    with open(f"{root}/info/{info_file}") as file:
        lines = file.readlines()
    paths = []
    labels = []
    if val:
        root = f"{root}/val_images_256"
    for line in lines:
        p, l = line.rstrip().split()
        l = int(l)
        if l < 50:
            paths.append(f"{root}/{p}")
            labels.append(l)
    return paths, labels


def prepare_dataset(dataset_name, noise, noise_mode, val_size):
    dataset_items = dict()
    dataset_items["train"] = dict()
    dataset_items["val"] = dict()
    dataset_items["test"] = dict()
    if dataset_name == "cifar10":
        dataset_items["num_classes"] = 10
        data_folder = "./dataset"
        trainset = torchvision.datasets.CIFAR10(
            root=data_folder, train=True, download=True
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_folder, train=False, download=True
        )

    if dataset_name == "cifar100":
        dataset_items["num_classes"] = 100
        data_folder = "./dataset"
        trainset = torchvision.datasets.CIFAR100(
            root=data_folder, train=True, download=True
        )
        testset = torchvision.datasets.CIFAR100(
            root=data_folder, train=False, download=True
        )
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        if val_size == 0:
            trainset_data, valset_data, trainset_target, valset_target = (
                trainset.data,
                [],
                trainset.targets,
                [],
            )
        else:
            (
                trainset_data,
                valset_data,
                trainset_target,
                valset_target,
            ) = train_test_split(
                trainset.data,
                trainset.targets,
                test_size=val_size,
                random_state=42,
                stratify=trainset.targets,
            )

        del trainset

        dataset_items["train"]["images"] = trainset_data
        dataset_items["train"]["clean_labels"] = torch.tensor(trainset_target)
        dataset_items["train"]["noisy_labels"] = apply_noise(
            torch.tensor(trainset_target),
            noise=noise,
            noise_mode=noise_mode,
            dataset_name=dataset_name,
        )

        dataset_items["val"]["images"] = valset_data
        dataset_items["val"]["clean_labels"] = torch.tensor(valset_target)
        dataset_items["val"]["noisy_labels"] = apply_noise(
            torch.tensor(valset_target),
            noise=noise,
            noise_mode="sym",
            dataset_name=dataset_name,
        )
        dataset_items["val"]["transforms"] = transforms.Compose(
            [
                Image.fromarray,
                transforms.ToTensor(),
                transforms.Normalize(*NORMALIZATION),
            ]
        )

        dataset_items["test"]["images"] = testset.data
        dataset_items["test"]["clean_labels"] = torch.tensor(testset.targets)
        dataset_items["test"]["noisy_labels"] = apply_noise(
            torch.tensor(testset.targets),
            noise=noise,
            noise_mode="sym",
            dataset_name=dataset_name,
        )
        dataset_items["test"]["transforms"] = transforms.Compose(
            [
                Image.fromarray,
                transforms.ToTensor(),
                transforms.Normalize(*NORMALIZATION),
            ]
        )

    elif dataset_name == "cloth":
        dataset_items["num_classes"] = 14
        data_folder = "./dataset/Cloth1M/"
        train_paths, train_noisy_labels = get_cloth1m_paths_labels(
            map_path="noisy_label_kv.txt",
            keys_path="noisy_train_key_list.txt",
            root=data_folder,
        )
        val_paths, val_clean_labels = get_cloth1m_paths_labels(
            map_path="clean_label_kv.txt",
            keys_path="clean_val_key_list.txt",
            root=data_folder,
        )
        test_paths, test_clean_labels = get_cloth1m_paths_labels(
            map_path="clean_label_kv.txt",
            keys_path="clean_test_key_list.txt",
            root=data_folder,
        )

        dataset_items["train"]["images"] = train_paths
        dataset_items["train"]["clean_labels"] = torch.tensor(train_noisy_labels)
        dataset_items["train"]["noisy_labels"] = torch.tensor(train_noisy_labels)

        dataset_items["val"]["images"] = val_paths
        dataset_items["val"]["clean_labels"] = torch.tensor(val_clean_labels)
        dataset_items["val"]["noisy_labels"] = torch.tensor(val_clean_labels)

        dataset_items["test"]["images"] = test_paths
        dataset_items["test"]["clean_labels"] = torch.tensor(test_clean_labels)
        dataset_items["test"]["noisy_labels"] = torch.tensor(test_clean_labels)

        dataset_items["train"]["transforms"] = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset_items["val"]["transforms"] = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset_items["test"]["transforms"] = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif dataset_name == "webvision":
        dataset_items["num_classes"] = 50
        data_folder = "./dataset/webvision"

        train_paths, train_noisy_labels = get_webvision_paths_labels(
            info_file="train_filelist_google.txt", root=data_folder
        )
        val_paths, val_clean_labels = get_webvision_paths_labels(
            info_file="val_filelist.txt", root=data_folder, val=True
        )

        dataset_items["train"]["images"] = train_paths
        dataset_items["train"]["clean_labels"] = torch.tensor(train_noisy_labels)
        dataset_items["train"]["noisy_labels"] = torch.tensor(train_noisy_labels)

        dataset_items["val"]["images"] = val_paths
        dataset_items["val"]["clean_labels"] = torch.tensor(val_clean_labels)
        dataset_items["val"]["noisy_labels"] = torch.tensor(val_clean_labels)

        dataset_items["test"]["images"] = []
        dataset_items["test"]["clean_labels"] = []
        dataset_items["test"]["noisy_labels"] = []

        dataset_items["train"]["transforms"] = transforms.Compose(
            [
                transforms.RandomCrop((227, 227)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset_items["val"]["transforms"] = transforms.Compose(
            [
                transforms.CenterCrop((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset_items["test"]["transforms"] = transforms.Compose(
            [
                transforms.CenterCrop((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        raise ValueError()

    return dataset_items


def create_train_sampler(noisy_labels, num_classes):
    N = len(noisy_labels)
    weights = [None] * N
    class_count = [0] * num_classes
    for i in range(N):
        label = noisy_labels[i]
        class_count[label.item()] += 1
    for i in range(N):
        label = noisy_labels[i]
        weights[i] = 1 / class_count[label.item()]

    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), N)
    return sampler


def create_dataloaders(
    dataset_name,
    batch_size,
    num_workers,
    noise,
    train_transform,
    noise_mode,
    val_size,
    with_sampler=False,
):
    dataset_items = prepare_dataset(dataset_name, noise, noise_mode, val_size)
    # TODO: move this to prepare dataset
    if "transforms" not in dataset_items["train"]:
        dataset_items["train"]["transforms"] = train_transform
    train_set = NoisyDataset(**dataset_items["train"])
    val_set = NoisyDataset(**dataset_items["val"])
    test_set = NoisyDataset(**dataset_items["test"])
    print(len(train_set), len(val_set), len(test_set))

    train_sampler = None
    if with_sampler:
        print("Using weighted sampler")
        train_sampler = create_train_sampler(
            dataset_items["train"]["noisy_labels"], dataset_items["num_classes"]
        )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True if train_sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    return loaders, dataset_items["num_classes"]


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
