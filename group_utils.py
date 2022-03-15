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
        """_summary_

        Args:
            total_len (_type_): _description_
        """
        self.total_len = total_len
        self.groups: List[List[int]] = None
        self.count = 0

    def next_group(self, epoch) -> List[int]:
        # return self.groups[epoch % len(self.groups)]
        self.count += 1
        return self.groups[self.count % len(self.groups)]
