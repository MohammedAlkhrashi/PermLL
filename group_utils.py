from typing import List

import torch.nn as nn


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
    def __init__(self, total_len) -> None:
        self.total_len = total_len
        self.groups: List[List[int]] = [[i] for i in range(total_len)]
        self.count = 0

    def next_group(self, epoch) -> List[int]:
        # next_group_index = self.count % len(self.groups)
        # self.count += 1
        # return self.groups[next_group_index]
        return [item for sublist in self.groups for item in sublist]

