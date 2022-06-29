import random
from typing import List

import torchvision
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import KLDivLoss, L1Loss, MSELoss
from torch.optim import SGD, Adam

from model import GroupModel
from PreResNet import PreActResNet18, PreActResNet34
from resnet import ResNet18, ResNet34


class GroupLoss(nn.Module):
    def __init__(self, criterion: nn.Module, equalize_losses):
        super().__init__()
        self.criterion: nn.Module = criterion
        self.equalize_losses = equalize_losses

    def forward(self, logits: List, target):
        if isinstance(self.criterion, (MSELoss, L1Loss, KLDivLoss)):
            num_classes = logits[0].size(1)
            target = F.one_hot(target, num_classes=num_classes)
            target = target.float()
            assert logits[0].shape == target.shape

        loss1 = 0
        loss2 = 0 
        for logit in logits:
            if isinstance(logit,tuple) and len(logit) == 2:
                loss1 += self.criterion(logit[0], target[0]).mean()
                loss2 += self.criterion(logit[1], target[1]).mean()
            else:
                loss1 += self.criterion(logit[0], target[0]).mean()
        return (loss1,loss2), loss1.sort(dim=0)[0]


class NLLSmoothing(nn.Module):
    """based on: https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631"""

    def __init__(self, smoothing=0.0, dim=-1, reduction="mean"):
        super(NLLSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
        self.reduction = reduction

    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            num_classes = pred.size(-1)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        loss = torch.sum(-true_dist * pred, dim=self.dim)
        return torch.mean(loss) if self.reduction == "mean" else loss


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


def model_from_name(model_name, num_classes, pretrained):
    if model_name == "resnet18":
        return ResNet18(num_classes)
    elif model_name == "resnet34":
        return ResNet34(num_classes)
    elif model_name == "resnet50":
        # for Clothing-1M
        model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        in_ = model.fc.in_features
        model.fc = nn.Linear(in_, num_classes)
        return model
    elif model_name == "preactresnet18":
        return PreActResNet18(num_classes)
    elif model_name == "preactresnet34":
        return PreActResNet34(num_classes)
    else:
        print("WARNING: Model loaded from timm,ignore if expected")
        return timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )


def create_group_model(
    num_of_networks,
    pretrained,
    num_classes,
    dataset_targets,
    init_max_prob,
    avg_before_perm,
    model_name="resnet18",
    disable_perm=False,
    softmax_temp=1,
    logits_softmax_mode=False,
):

    models = nn.ModuleList()
    model_names_list = model_name.split(",")
    repeat_factor = num_of_networks / len(model_names_list)
    if not repeat_factor.is_integer():
        raise ValueError(
            "num_of_networks is not divisiable by the length of the: specific model names entered"
        )
    model_names_list = model_names_list * int(repeat_factor)
    print(f"Models in GroupModel: {model_names_list}")
    print(f"Number of classes: {num_classes}")
    for model_name in model_names_list:
        model = model_from_name(model_name, num_classes, pretrained)
        models.append(model)
    group_model = GroupModel(
        models,
        num_classes,
        dataset_targets,
        init_max_prob,
        avg_before_perm,
        disable_perm,
        softmax_temp=softmax_temp,
        logits_softmax_mode=logits_softmax_mode,
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
