from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from callbacks import Callback, permuted_samples
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
        grad_clip,
        group_picker: GroupPicker,
        num_permutation_limit,
        equalize_losses,
        callbacks: List[Callback] = [],
        gpu_num="0",
    ) -> None:
        self.model: GroupModel = model
        self.optimizer: GroupOptimizer = optimizer
        self.criterion: nn.Module = GroupLoss(criterion, equalize_losses)
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.num_permutation_limit = num_permutation_limit

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
        loss, all_losses = self.criterion(output, batch["noisy_label"])
        if not val_step:
            loss.backward()
            if self.grad_clip != -1:
                # Grad clipping currently only works for one network.
                nn.utils.clip_grad_value_(
                    self.model.models[0].parameters(), self.grad_clip
                )
            self.optimizer.step()

        metrics = {
            "batch": batch,
            "loss": loss,
            "output": output,
            "unpermuted_logits": unpermuted_logits,
            "sample_index": batch["sample_index"],
            "alpha_matrix": self.model.perm_model.alpha_matrix,
            "all_clean_labels": self.train_loader.dataset.original_labels,
            "all_noisy_labels": self.train_loader.dataset.noisy_labels,
            "all_losses": all_losses,
        }
        return metrics

    def one_epoch(self, epoch, val_epoch=False):
        loader = self.val_loader if val_epoch else self.train_loader
        name = "val" if val_epoch else "train"

        for batch in tqdm(
            loader, desc=f"Running {name.capitalize()}, Epoch: {epoch}", leave=True
        ):
            metrics = self.step(batch=batch, epoch=epoch, val_step=val_epoch)
            for callback in self.callbacks:
                callback.on_step_end(metrics, name)

        for callback in self.callbacks:
            callback.on_epoch_end(metrics, epoch, name)

        return metrics

    def start(self):
        self.model.to(self.device)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            metrics = self.one_epoch(epoch, val_epoch=False)
            with torch.no_grad():
                self.model.eval()
                self.one_epoch(epoch, val_epoch=True)

            for callback in self.callbacks:
                if callback.early_stop():
                    print("Stopping: No improvment for while.")
                    return
            _, _, num_permuted_samples = permuted_samples(metrics)
            if (
                num_permuted_samples > self.num_permutation_limit
                and self.num_permutation_limit != -1
            ):
                return

        for callback in self.callbacks:
            callback.on_training_end(metrics)
