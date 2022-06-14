import torch
import torch.nn as nn


# def stable_softmax(tensor):
#     pass
def stable_log_softmax(tensor):
    # t_max, _= torch.max(tensor,dim=1)
    # t_max = t_max[:,None]
    # exp = torch.exp(tensor - t_max)
    # log_exp= torch.log(exp)
    # log_sum_exp_bot = torch.logsumexp(tensor,dim=1)[:,None]

    # return (-log_sum_exp_bot + t_max) + log_exp
    t_max, _ = torch.max(tensor, dim=1)
    t_max = t_max[:, None]
    exp = tensor - t_max
    log_sum_exp_bot = torch.logsumexp(exp, dim=1)[:, None]
    return exp - log_sum_exp_bot

def log_perm_softmax(P, a, f_x):
    M = torch.max(a+f_x).unsqueeze(-1)
    M = 0
    log_sum_exp_alphas = torch.logsumexp(a, dim=1).unsqueeze(-1)
    log_sum_exp_f_x = torch.logsumexp(f_x, dim=1).unsqueeze(-1)
    rep_alphas = a.repeat_interleave(a.size(1)**2).reshape(P.shape)
    rep_alphas = rep_alphas + f_x.reshape(-1,1,1,a.size(1))
    normalized = rep_alphas - M
    exp_normalized = torch.exp(normalized)
    scaled_perms = exp_normalized * P
    sum_perm = torch.sum(scaled_perms, dim=1)
    print("sum perm: ",sum_perm)
    perm_x_ones = torch.sum(sum_perm,dim=-1)
    return torch.log(perm_x_ones) + M - (log_sum_exp_alphas + log_sum_exp_f_x)
    
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
                permuted_logits = self.perm_model(logits, target, sample_index, self.logits_softmax_mode)
                permuted_logits = self.apply_post_perm_op(permuted_logits)
                all_permuted_logits.append(permuted_logits)
                all_unpermuted_logits.append(logits)
            return all_permuted_logits, all_unpermuted_logits

    def apply_pre_perm_op(self, logits):
        print('PRE: logits before:')
        print(logits)
        if self.logits_softmax_mode == "log_softmax":
            print("log_softmax_pre")
            logits =  torch.log_softmax(logits, dim=1)
        elif self.logits_softmax_mode == "softmax":
            print("softmax_pre")
            logits = torch.softmax(logits, dim=1)
        elif self.logits_softmax_mode == "stable_softmax":
            print("stable_softmax_pre")
            logits = stable_log_softmax(logits)
        else:
            print("default_before")

        print('\nPRE: logits after:')
        print(logits)
        return logits

    def apply_post_perm_op(self, permuted_logits):
        print('POST: logits before:')
        print(permuted_logits)
        if self.logits_softmax_mode == "log_softmax":
            print("log_softmax_post")
        elif self.logits_softmax_mode == "softmax":
            print("softmax_after")
            permuted_logits =  torch.log(permuted_logits)
        elif self.logits_softmax_mode == "stable_softmax":
            print("stable_softmax_post")
            pass
        else:
            print("default_after")
        print('\nPOST: logits after:')
        print(permuted_logits)
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
            return logits

        perm = self.all_perm[target]
        perm = perm.to(self.alpha_matrix.device)
        alpha = self.alpha_matrix[sample_index] / self.softmax_temp
        if logits_softmax_mode == "log_perm_softmax":
            return log_perm_softmax(perm,alpha,logits)
        else:
            permutation_matrices = (
                self.softmax(alpha.unsqueeze(-1).unsqueeze(-1)) * perm
            ).sum(1)
            permuted_logits = torch.matmul(
                permutation_matrices, logits.unsqueeze(-1)
            ).squeeze(-1)
            return permuted_logits

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
