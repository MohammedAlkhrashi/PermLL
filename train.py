from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from callbacks import Callback
from group_utils import GroupLoss, GroupOptimizer, GroupPicker
from model import GroupModel


class TrainPermutation:
    def __init__(
        self,
        model: GroupModel,
        optimizer,
        train_loader,
        val_loader,
        epochs,
        criterion,
        group_picker: GroupPicker,
        callbacks: List[Callback] = [],
        gpu_num="0",
    ) -> None:
        self.model: GroupModel = model
        self.optimizer: GroupOptimizer = optimizer
        self.criterion: nn.Module = GroupLoss(criterion)
        self.epochs = epochs

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(
            f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
        )
        self.group_picker = group_picker
        self.callbacks: List[Callback] = callbacks

    def step(self, batch, epoch, val_step=False):
        batch = {key: value.to(self.device) for key, value in batch.items()}
        next_group: List[int] = self.group_picker.next_group(val_step)

        self.optimizer.zero_grad()
        output, unpermuted_logits = self.model(
            batch["image"],
            target=batch["noisy_label"],
            sample_index=batch["sample_index"],
            network_indices=next_group,
        )
        loss = self.criterion(output, batch["noisy_label"])
        if not val_step:
            loss.backward()
            self.optimizer.step()

        metrics = {
            "batch": batch,
            "loss": loss,
            "output": output,
            "unpermuted_logits": unpermuted_logits,
            "alpha_matrix": self.model.perm_model.alpha_matrix,
            "all_clean_labels": self.train_loader.dataset.original_labels,
            "all_noisy_labels": self.train_loader.dataset.noisy_labels,
        }
        return metrics

    def one_epoch(self, epoch, val_epoch=False):
        loader = self.val_loader if val_epoch else self.train_loader
        name = "val" if val_epoch else "train"

        for batch in tqdm(loader, desc=f"Running {name.capitalize()} Epoch"):
            metrics = self.step(batch=batch, epoch=epoch, val_step=val_epoch)

            for callback in self.callbacks:
                callback.on_step_end(metrics, name)

        for callback in self.callbacks:
            callback.on_epoch_end(metrics, epoch, name)

    def start(self):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            self.model.train()
            self.one_epoch(epoch, val_epoch=False)
            self.model.eval()
            with torch.no_grad():
                self.one_epoch(epoch, val_epoch=True)
