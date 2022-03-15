from typing import List
import torch.nn as nn


class GroupModel(nn.Module):
    def __init__(self, models: List[nn.Module], num_classes: int) -> None:
        self.models = models
        self.perm_model = PermutationModel(num_classes)
        super().__init__()

    def forward(self, x, target, sample_index, network_indices):
        cur_models = self.models[network_indices]
        outputs = []
        for model in cur_models:
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
        class_perm = self.perm_list[target]
        alphas = self.alphas[sample_index]
        permutation_matrix = None  # TODO: from alphas and class_prem
        permuted_logits = None  # TODO: from permutation_matrix and x
        return permuted_logits
