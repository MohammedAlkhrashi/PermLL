import timm
import torch
import torch.nn as nn


class GroupModel(nn.Module):
    def __init__(
        self,
        models: nn.ModuleList,
        num_classes: int,
        dataset_targets,
        perm_init_value,
        disable_perm=False,
    ) -> None:
        super().__init__()
        self.models = models
        self.perm_model = PermutationModel(
            num_classes,
            dataset_targets,
            perm_init_value,
            disable_module=disable_perm,
        )

    def forward(self, x, target, sample_index, network_indices):
        outputs = []
        all_unpermuted_logits = []
        for index in network_indices:
            model = self.models[index]
            logits = model(x)
            permuted_logits = self.perm_model(logits, target, sample_index)
            outputs.append(permuted_logits)
            all_unpermuted_logits.append(logits)
        return outputs, all_unpermuted_logits

    def __len__(self):
        return len(self.models)


class PermutationModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dataset_targets: list,
        perm_init_value: float,
        disable_module=False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        num_train_samples = len(dataset_targets)
        self.softmax = nn.Softmax(1)
        self.alpha_matrix = nn.Parameter(
            torch.zeros(num_train_samples, num_classes), requires_grad=False
        )
        self.alpha_matrix.scatter_(
            1, torch.tensor(dataset_targets).unsqueeze(1), perm_init_value
        )
        self.alpha_matrix.requires_grad = True
        self.all_perm = self.create_all_perm()

        self.disable_module = disable_module
        if self.disable_module:
            print("*" * 100)
            print(
                "WARNING: Permutation model is disabled, it is now equivelent to the identity module"
            )
            print("*" * 100)

    def forward(self, logits, target, sample_index):
        if not self.training or self.disable_module:
            return logits
        perm = self.all_perm[target]
        perm = perm.to(self.alpha_matrix.device)
        alpha = self.alpha_matrix[sample_index]
        permutation_matrices = (
            self.softmax(alpha.unsqueeze(-1).unsqueeze(-1)) * perm
        ).sum(1)
        permuted_logits = torch.matmul(
            permutation_matrices, logits.unsqueeze(-1)
        ).squeeze(-1)
        return permuted_logits

    def create_all_perm(self):
        classes_list = list(range(self.num_classes))
        I = torch.eye(self.num_classes)
        perm_list = []
        for idx1 in range(self.num_classes):
            one_class_perm_list = []
            for idx2 in range(self.num_classes):
                l = classes_list.copy()
                l = swap_positions(l, idx1, idx2)
                perm = I[l]
                one_class_perm_list.append(perm)
            one_class_perm = torch.stack(one_class_perm_list)
            perm_list.append(one_class_perm)
        perm = torch.stack(perm_list)
        return perm


def swap_positions(list_, pos1, pos2):
    list_[pos1], list_[pos2] = list_[pos2], list_[pos1]
    return list_
