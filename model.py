from typing import List
import torch.nn as nn
import timm


class GroupModel(nn.Module):
    def __init__(self, models: nn.ModuleList, num_classes: int) -> None:
        super().__init__()
        self.models = models
        self.perm_model = PermutationModel(num_classes)

    def forward(self, x, target, sample_index, network_indices):
        outputs = []
        for index in network_indices:
            model = self.models[index]
            logits = model(x)
            permuted_logits = self.perm_model(logits, target, sample_index)
            outputs.append(permuted_logits)
        return outputs

    def __len__(self):
        return len(self.models)


class PermutationModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.perm_list = None  # TODO
        self.alphas = None  # TODO

    def forward(self, x, target, sample_index):
        return x
        if not self.training:
            return x
        class_perm = self.perm_list[target]
        alphas = self.alphas[sample_index]
        permutation_matrix = None  # TODO: from alphas and class_prem
        permuted_logits = None  # TODO: from permutation_matrix and x
        return permuted_logits


def create_group_model(num_of_networks, pretrained, num_classes, model_name="resnet18"):
    models = nn.ModuleList()
    for i in range(num_of_networks):
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
        models.append(model)
    group_model = GroupModel(models, num_classes)
    return group_model

