import torch
import torch.nn as nn
import numpy as np


def log_perm_softmax_stable(P, alphas, logits):
    log_sum_exp_alphas = torch.logsumexp(alphas, dim=1).unsqueeze(-1)
    log_sum_exp_f_x = torch.logsumexp(logits, dim=1).unsqueeze(-1)

    alphas_expanded = alphas.repeat_interleave(alphas.size(1) ** 2).reshape(P.shape)
    exponents = alphas_expanded + logits.reshape(-1, 1, 1, alphas.size(1))
    masked_exponents = exponents * P

    neg_infs = torch.full(masked_exponents.shape, -np.inf, device=logits.device)
    final_exponents = torch.where(masked_exponents != 0, masked_exponents, neg_infs)
    log_sum_exp_over_all_rows = torch.logsumexp(final_exponents, dim=(1, 3))

    log_permuted_softmax_logits = log_sum_exp_over_all_rows - (
        log_sum_exp_alphas + log_sum_exp_f_x
    )  # =log(permutation(softmax(logits)))
    return log_permuted_softmax_logits


def perm_logits(logits, perm, alpha):
    permutation_matrices = (
        torch.softmax(alpha.unsqueeze(-1).unsqueeze(-1), dim=1) * perm
    ).sum(1)
    permuted_logits = torch.matmul(permutation_matrices, logits.unsqueeze(-1)).squeeze(
        -1
    )
    return permuted_logits


class GroupModel(nn.Module):
    def __init__(
        self,
        models: nn.ModuleList,
        num_classes: int,
        dataset_targets,
        perm_init_value,
        avg_before_perm,
        disable_perm=False,
        softmax_temp=1,
        logits_softmax_mode=False,
    ) -> None:
        super().__init__()
        self.models = models
        self.avg_before_perm = avg_before_perm
        self.perm_model = PermutationModel(
            num_classes,
            dataset_targets,
            perm_init_value,
            disable_module=disable_perm,
            softmax_temp=softmax_temp,
        )
        self.logits_softmax_mode = logits_softmax_mode

    def forward(self, x, target, sample_index, network_indices):
        all_permuted_logits = []
        all_unpermuted_logits = []
        if self.avg_before_perm:
            for index in network_indices:
                model = self.models[index]
                logits = model(x)
                if self.logits_softmax_mode:
                    logits = torch.log_softmax(logits, dim=1)
                all_unpermuted_logits.append(logits)
            unpermuted_logits = torch.stack(all_unpermuted_logits)
            # unpermuted_logits = normalize(unpermuted_logits, dim=2)
            unpermuted_logits = unpermuted_logits.mean(0)
            permuted_logits = self.perm_model(unpermuted_logits, target, sample_index)
            all_permuted_logits.append(permuted_logits)
            return all_permuted_logits, [unpermuted_logits]
        else:
            for index in network_indices:
                model = self.models[index]
                logits = model(x)
                logits = self.apply_pre_perm_op(logits)
                permuted_logits = self.perm_model(
                    logits, target, sample_index, self.logits_softmax_mode
                )
                permuted_logits = self.apply_post_perm_op(permuted_logits)
                all_permuted_logits.append(permuted_logits)
                all_unpermuted_logits.append(logits)
            return all_permuted_logits, all_unpermuted_logits

    def apply_pre_perm_op(self, logits):
        if self.logits_softmax_mode == "log_softmax":
            logits = torch.log_softmax(logits, dim=1)
        elif self.logits_softmax_mode == "softmax":
            logits = torch.softmax(logits, dim=1)
        else:
            pass
        return logits

    def apply_post_perm_op(self, permuted_logits):
        if self.logits_softmax_mode == "log_softmax":
            pass
        elif self.logits_softmax_mode == "softmax":
            pass
        else:
            pass
        return permuted_logits

    def __len__(self):
        return len(self.models)


class PermutationModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dataset_targets: list,
        perm_init_value: float,
        disable_module=False,
        softmax_temp=1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_train_samples = len(dataset_targets)
        self.softmax = nn.Softmax(1)
        self.perm_init_value = perm_init_value

        self.alpha_matrix = self.create_alpha_matrix(dataset_targets)
        self.all_perm = self.create_all_perm()

        self.softmax_temp = softmax_temp
        self.disable_module = disable_module
        if self.disable_module:
            print("*" * 100)
            print(
                "WARNING: Permutation model is disabled, it is now equivelent to the identity module"
            )
            print("*" * 100)

    def forward(self, logits, target, sample_index, logits_softmax_mode=None):
        if not self.training or self.disable_module:
            return (
                torch.log_softmax(logits, dim=1)
                if logits_softmax_mode == "log_perm_softmax"
                else logits
            )
        perm = self.all_perm[target]
        perm = perm.to(self.alpha_matrix.device)
        alpha = self.alpha_matrix[sample_index] / self.softmax_temp

        if logits_softmax_mode == "log_perm_softmax":
            return log_perm_softmax_stable(perm, alpha, logits)
        else:
            return perm_logits(logits, perm, alpha)

    def create_alpha_matrix(self, targets):
        alpha_matrix = nn.Parameter(
            torch.zeros(self.num_train_samples, self.num_classes), requires_grad=False
        )
        alpha_matrix.scatter_(
            1, torch.tensor(targets).unsqueeze(1), self.perm_init_value
        )
        alpha_matrix.requires_grad = True
        return alpha_matrix

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
