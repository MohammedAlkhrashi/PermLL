from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from group_utils import GroupLoss, GroupPicker
from model import GroupModel


def handle_stats(model, batch, output, loss, epoch, stats_name):
    """
    handle stats, can be converted into a class or a hook. 
    """
    pass


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
        self.model: GroupModel = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.criterion: nn.Module = GroupLoss(criterion)
        self.epochs = epochs

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(
            f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
        )
        self.group_picker = GroupPicker(total_len=len(self.model))

    def step(self, batch, epoch, val_step=False):
        batch = {key: value.to(self.device) for key, value in batch.items()}
        self.optimizer.zero_grad()
        next_group: List[int] = self.group_picker.next_group(epoch)
        output = self.model(
            batch["image"],
            target=batch["noisy_label"],
            sample_index=batch["sample_index"],
            network_indices=next_group,
        )
        loss = self.criterion(output, batch["noisy_label"])
        if not val_step:
            loss.backward()
            self.optimizer.step()

        metrics = {"batch": batch, "loss": loss, "output": output}
        return metrics

    def log_stats(
        self, running_loss, noisy_running_correct, clean_running_correct, total, epoch, set
    ):
        wandb.log({f"{set}_clean_accuracy": clean_running_correct / total})
        wandb.log({f"{set}_noisy_accuracy": noisy_running_correct / total})
        wandb.log({f"{set}_noisy_loss": running_loss / total})
        print(f'{set}_clean acc = {clean_running_correct / total}')
        print(f'{set}_noisy acc = {noisy_running_correct / total}')
    def start(self):
        self.model.to(self.device)

        for epoch in range(self.epochs):
            running_loss = 0
            noisy_running_correct = 0
            clean_running_correct = 0
            total = 0

            self.model.train()
            for batch in tqdm(self.train_loader):
                metrics = self.step(batch=batch, epoch=epoch)

                # TODO: quick and dirty solution for metrics, clean up.
                running_loss += metrics["loss"]
                _, predicted = torch.max(metrics["output"][0].detach(), 1)
                noisy_running_correct += (
                    (predicted == metrics["batch"]["noisy_label"]).sum().item()
                )
                clean_running_correct += (
                    (predicted == metrics["batch"]["true_label"]).sum().item()
                )
                total += batch["noisy_label"].size(0)


            self.log_stats(
                running_loss,
                noisy_running_correct,
                clean_running_correct,
                total,
                epoch,
                set="train",
            )

            running_loss = 0
            noisy_running_correct = 0
            clean_running_correct = 0
            total = 0

            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(self.val_loader):
                    metrics = self.step(batch=batch, epoch=epoch, val_step=True)

                    # TODO: quick and dirty solution for metrics, clean up.
                    running_loss += metrics["loss"]
                    _, predicted = torch.max(metrics["output"][0].detach(), 1)
                    noisy_running_correct += (
                        (predicted == metrics["batch"]["noisy_label"]).sum().item()
                    )
                    clean_running_correct += (
                        (predicted == metrics["batch"]["true_label"]).sum().item()
                    )
                    total += batch["noisy_label"].size(0)

            self.log_stats(
                running_loss,
                noisy_running_correct,
                clean_running_correct,
                total,
                epoch,
                set="val",
            )

