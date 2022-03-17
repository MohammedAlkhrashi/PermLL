import torch
import wandb


class Callback:
    def on_step_end(self, metrics, name):
        raise NotImplementedError

    def on_epoch_end(self, metrics, epoch, name):
        raise NotImplementedError


class CallbackNoisyStatistics(Callback):
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.running_loss = 0
        self.noisy_running_correct = 0
        self.clean_running_correct = 0
        self.total = 0

    def on_step_end(self, metrics, name):
        self.running_loss += metrics["loss"]
        _, predicted = torch.max(metrics["output"][0].detach(), 1)
        _, prediction_before_perm = torch.max(metrics["unpermuted_logits"][0].detach(), 1)

        self.noisy_running_correct += (
            (predicted == metrics["batch"]["noisy_label"]).sum().item()
        )
        self.clean_running_correct += (
            (prediction_before_perm == metrics["batch"]["true_label"]).sum().item()
        )
        self.total += metrics["batch"]["noisy_label"].size(0)

    def on_epoch_end(self, epoch, name):
        self.log_stats(name, epoch)
        self.reset()

    def log_stats(self, name, epoch):
        wandb.log({f"{name}_clean_accuracy": self.clean_running_correct / self.total})
        wandb.log({f"{name}_noisy_accuracy": self.noisy_running_correct / self.total})
        wandb.log({f"{name}_noisy_loss": self.running_loss / self.total})
        print(f'{set}_clean acc = {self.clean_running_correct / self.total}')
        print(f'{set}_noisy acc = {self.noisy_running_correct / self.total}')