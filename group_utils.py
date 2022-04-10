import random
from typing import List

import timm
import torch.nn as nn
from torch.optim import SGD, Adam

from model import GroupModel
from resnet import ResNet18, ResNet34
from PreResNet import PreActResNet18, PreActResNet34


class GroupLoss(nn.Module):
    def __init__(self, criterion: nn.Module):
        super().__init__()
        self.criterion: nn.Module = criterion

    def forward(self, logits: List, target):
        # TODO: check stack method instead
        loss = 0
        for logit in logits:
            loss += self.criterion(logit, target)
        return loss


class GroupPicker:
    def __init__(self, networks_per_group, num_groups, change_every) -> None:
        self.networks_per_group = networks_per_group
        self.num_groups = num_groups
        self.groups: List[List[int]] = self.create_new_groups()
        self.change_every = change_every
        self.count = 0
        self.cur_group_index = 0

    def create_new_groups(self):
        total_networks = self.networks_per_group * self.num_groups
        indices = list(range(total_networks))
        random.shuffle(indices)
        groups = []
        for i in range(0, total_networks, self.networks_per_group):
            group = indices[i : i + self.networks_per_group]
            groups.append(group)
        return groups

    def all_groups(self):
        return [item for sublist in self.groups for item in sublist]

    def next_group(self, val_step=False) -> List[int]:
        if val_step:
            return self.all_groups()

        if self.count % self.change_every == 0:
            self.cur_group_index = (self.cur_group_index + 1) % len(self.groups)
        next_group = self.groups[self.cur_group_index]
        self.count += 1
        return next_group


class GroupOptimizer:
    def __init__(self, networks_optimizer, permutation_optimizer) -> None:
        self.network_optimizer = networks_optimizer
        self.permutation_optimizer = permutation_optimizer

    def step(self):
        self.network_optimizer.step()
        self.permutation_optimizer.step()

    def zero_grad(self):
        self.network_optimizer.zero_grad()
        self.permutation_optimizer.zero_grad()


def create_group_optimizer(
    model,
    networks_optim_choice,
    networks_lr,
    permutation_lr,
    weight_decay,
    momentum,
    perm_optimizer,
    perm_momentum,
):

    networks_params = []
    permutations_params = []
    for name, param in model.named_parameters():
        if name.startswith("perm_model"):
            permutations_params.append(param)
        else:
            networks_params.append(param)
    assert len(permutations_params) != 0

    if networks_optim_choice == "sgd":
        networks_optimizer = SGD(
            networks_params,
            lr=networks_lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

    elif networks_optim_choice == "adam":
        networks_optimizer = Adam(
            networks_params, lr=networks_lr, weight_decay=weight_decay
        )

    else:
        raise ValueError()

    if perm_optimizer == "sgd":
        permutation_optimizer = SGD(
            permutations_params, lr=permutation_lr, momentum=perm_momentum
        )
    elif perm_optimizer == "adam":
        permutation_optimizer = Adam(permutations_params, lr=permutation_lr)
    else:
        raise ValueError()

    return GroupOptimizer(
        networks_optimizer=networks_optimizer,
        permutation_optimizer=permutation_optimizer,
    )


def model_from_name(model_name):
    if model_name == "resnet18":
        return ResNet18()
    elif model_name == "resnet34":
        return ResNet34()
    elif model_name == "preactresnet18":
        return PreActResNet18()
    elif model_name == "preactresnet34":
        return PreActResNet34()
    else:
        print("WARNING: Model loaded from timm,ignore if expected")
        return timm.create_model(model_name, pretrained=False, num_classes=10)


def create_group_model(
    num_of_networks,
    pretrained,
    num_classes,
    dataset_targets,
    perm_init_value,
    avg_before_perm,
    model_name="resnet18",
    disable_perm=False,
):
    models = nn.ModuleList()
    for _ in range(num_of_networks):
        model = model_from_name(model_name)
        models.append(model)
    group_model = GroupModel(
        models,
        num_classes,
        dataset_targets,
        perm_init_value,
        avg_before_perm,
        disable_perm,
    )
    return group_model


def test_group_picker():
    def test_picker(networks_per_group, num_groups, change_every):
        group_picker = GroupPicker(
            networks_per_group=networks_per_group,
            num_groups=num_groups,
            change_every=change_every,
        )
        assert num_groups == len(
            group_picker.groups
        ), f"{num_groups} == {len(group_picker.groups)}"

        indices = []
        for group in group_picker.groups:
            assert len(group) == networks_per_group
            indices.extend(group)
        assert set(indices) == set(range(networks_per_group * num_groups))

        if num_groups != 1:
            count = 0
            prev_group = []
            for _ in range(change_every * 100):
                cur_group = group_picker.next_group()
                if count % change_every == 0:
                    assert cur_group != prev_group
                else:
                    assert cur_group == prev_group
                prev_group = cur_group
                count += 1
        assert len(group_picker.all_groups()) == num_groups * networks_per_group
        assert set(group_picker.all_groups()) == set(
            range(num_groups * networks_per_group)
        )

    test_picker(15, 10, 1)
    test_picker(3, 5, 5)
    test_picker(1, 100, 100)
    test_picker(100, 1, 1)
    test_picker(50, 50, 5)
    test_picker(6, 33, 77)


if __name__ == "__main__":
    test_group_picker()
