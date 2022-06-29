from typing import List
import numpy as np

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
        test_loader,
        epochs,
        criterion,
        grad_clip,
        group_picker: GroupPicker,
        num_permutation_limit,
        equalize_losses,
        callbacks: List[Callback] = [],
        gpu_num="0",
        mixup_alpha=-1,
    ) -> None:
        self.model: GroupModel = model
        self.optimizer: GroupOptimizer = optimizer
        self.criterion: nn.Module = GroupLoss(criterion, equalize_losses)
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.num_permutation_limit = num_permutation_limit
        self.mixup_alpha = mixup_alpha
        self.is_mixup = mixup_alpha != -1

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = torch.device(
            f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
        )
        self.group_picker = group_picker
        self.callbacks: List[Callback] = callbacks

    def step(self, batch, epoch, val_step=False):
        batch = {key: value.to(self.device) for key, value in batch.items()}
        next_group: List[int] = self.group_picker.next_group(val_step)

        (
            batch["image"],
            batch["noisy_label"],
            batch["true_label"],
            batch["sample_index"],
            lam,
        ) = mixup(
            self.mixup_alpha,
            batch["image"],
            batch["noisy_label"],
            batch["true_label"],
            batch["sample_index"],
            val_step,
            self.is_mixup,
        )

        self.optimizer.zero_grad()
        output, unpermuted_logits = self.model(
            batch["image"],
            target=batch["noisy_label"],
            sample_index=batch["sample_index"],
            network_indices=next_group,
        )
        batch["true_label"][0] if lam >= 0.5 else batch["true_label"][1]
        loss, all_losses = self.criterion(output, batch["noisy_label"])
        final_loss = lam * loss[0] + (1 - lam) * loss[1]
        if not val_step:
            final_loss.backward()
            self.perform_grad_clip()
            self.optimizer.step()

        # for stats
        output = output[0] if lam >= 0.5 else output[1]
        batch["true_label"] = (
            batch["true_label"][0] if lam >= 0.5 else batch["true_label"][1]
        )
        batch["noisy_label"] = (
            batch["noisy_label"][0] if lam >= 0.5 else batch["noisy_label"][1]
        )
        batch["sample_index"] = (
            batch["sample_index"][0] if lam >= 0.5 else batch["sample_index"][1]
        )
        metrics = {
            "batch": batch,
            "loss": final_loss,
            "output": output,
            "unpermuted_logits": unpermuted_logits,
            "sample_index": batch["sample_index"],
            "alpha_matrix": self.model.perm_model.alpha_matrix,
            "all_clean_labels": self.train_loader.dataset.original_labels,
            "all_noisy_labels": self.train_loader.dataset.noisy_labels,
            "all_losses": all_losses,
        }
        return metrics

    def perform_grad_clip(self):
        if self.grad_clip != -1:
            print("WARNING CLIPPING")
            # Grad clipping currently only works for one network.
            nn.utils.clip_grad_value_(self.model.models[0].parameters(), self.grad_clip)

    def one_epoch(self, epoch, loader, name, val_epoch=False):
        metrics = None
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
        self.model.perm_model.all_perm = self.model.perm_model.all_perm.to(self.device)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            metrics = self.one_epoch(
                epoch, loader=self.train_loader, name="train", val_epoch=False
            )
            with torch.no_grad():
                self.model.eval()
                self.one_epoch(
                    epoch, loader=self.val_loader, name="val", val_epoch=True
                )
                self.one_epoch(
                    epoch, loader=self.test_loader, name="test", val_epoch=True
                )

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


def mixup(alpha, image, noisy_target, clean_target, sample_index, val_step, is_mixup):
    """based on: https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py"""
    if val_step or not is_mixup:
        return image, (noisy_target,), (clean_target,), (sample_index,), 1

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = image.size()[0]
    index = torch.randperm(batch_size).to(image.device)

    mixed_image = lam * image + (1 - lam) * image[index, :]
    mixed_noisy_target = noisy_target, noisy_target[index]
    mixed_clean_target = clean_target, clean_target[index]
    mixed_sample_index = sample_index, sample_index[index]

    return mixed_image, mixed_noisy_target, mixed_clean_target, mixed_sample_index, lam
