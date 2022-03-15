from typing import List
import torch
import torch.nn as nn
import timm


class GroupModel(nn.Module):
    def __init__(self, models: nn.ModuleList, num_classes: int, dataset_targets) -> None:
        super().__init__()
        self.models = models
        self.perm_model = PermutationModel(num_classes, dataset_targets)

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
    def __init__(self, num_classes: int, dataset_targets: list, class_init_value: float=10) -> None:
        super().__init__()
        self.num_classes = num_classes
        num_train_samples = len(dataset_targets)
        self.softmax = nn.Softmax(1)
        self.alpha_matrix = nn.Parameter(torch.ones(num_train_samples, num_classes), requires_grad=False)
        self.alpha_matrix.scatter_(1, torch.tensor(dataset_targets).unsqueeze(1), class_init_value)
        self.alpha_matrix.requires_grad = True
        self.all_perm = self.create_all_perm()

    def forward(self, logits, target, sample_index):
        # return logits
        if not self.training:
            return logits
        perm = self.all_perm[target]
        perm = perm.to(self.alpha_matrix.device)
        alpha = self.alpha_matrix[sample_index]
        permutation_matrices = (self.softmax(alpha.unsqueeze(-1).unsqueeze(-1))*perm).sum(1)
        permuted_logits = torch.matmul(permutation_matrices, logits.unsqueeze(-1)).squeeze(-1)
        return permuted_logits

    def create_all_perm(self):
        classes_set = set(range(self.num_classes))
        I = torch.eye(self.num_classes)
        perm_list = []
        for target in classes_set:
            l1 = list(classes_set - set([target]))
            one_class_perm_list = []
            for idx in range(len(l1)+1):
                l2 = l1.copy()
                l2.insert(idx,target)
                perm = I[l2]
                one_class_perm_list.append(perm)
            one_class_perm = torch.stack(one_class_perm_list)
            perm_list.append(one_class_perm)
        perm = torch.stack(perm_list)
        return perm


def create_group_model(num_of_networks, pretrained, num_classes, dataset_targets, model_name="resnet18"):
    models = nn.ModuleList()
    for i in range(num_of_networks):
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
        models.append(model)
    group_model = GroupModel(models, num_classes, dataset_targets)
    return group_model

