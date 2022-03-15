from turtle import forward
from typing import List

import torch
import torch.nn as nn

from group_utils import GroupLoss, GroupPicker
from model import GroupModel


def handle_stats(model, image, label, output, loss, epoch, stats_name):
    """
    handle stats, can be converted into a class or a hook. 
    """
    raise NotImplemented


class TrainPermutation:
    def __init__(
        self,
        model: GroupModel,
        optimizer,
        train_loader,
        val_loader,
        epochs,
        criterion,
        gpu_num="0",
    ) -> None:
        self.model: nn.Module = model.to(self.device)
        self.optimizer: torch.optim.Optimizer = optimizer
        self.criterion: nn.Module = GroupLoss(criterion)
        self.epochs = epochs

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.group_picker = GroupPicker(total_len=len(self.model))
        self.device = torch.device(
            f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
        )

    def step(self, batch, epoch, val_step=False):
        batch = {key: value.to(self.device) for key, value in batch}
        self.optimizer.zero_grad()
        next_group: List[int] = self.group_picker.next_group(epoch)
        output = self.model(batch["image"], network_indices=next_group)
        loss = self.criterion(output, batch["label"])
        if not val_step:
            loss.backward()
            self.optimizer.step()

        # TODO: handle stats, inside a class or a list of hooks.
        handle_stats(
            self.model, batch["image"], batch["label"], output, loss, epoch, "train"
        )

    def start(self):
        for epoch in range(self.epochs):
            self.model.train()
            for batch in self.train_loader:
                self.step(batch=batch, epoch=epoch)

            self.model.eval()
            for batch in self.val_loader:
                self.step(batch=batch, epoch=epoch, val_step=True)

