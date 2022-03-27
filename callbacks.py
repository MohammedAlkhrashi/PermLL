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
        _, prediction_before_perm = torch.max(
            metrics["unpermuted_logits"][0].detach(), 1
        )

        self.noisy_running_correct += (
            (predicted == metrics["batch"]["noisy_label"]).sum().item()
        )
        self.clean_running_correct += (
            (prediction_before_perm == metrics["batch"]["true_label"]).sum().item()
        )
        self.total += metrics["batch"]["noisy_label"].size(0)

    def on_epoch_end(self, metrics, epoch, name):
        self.log_stats(name, epoch)
        self.reset()

    def log_stats(self, name, epoch):
        wandb.log({f"{name}_clean_accuracy": self.clean_running_correct / self.total})
        wandb.log({f"{name}_noisy_accuracy": self.noisy_running_correct / self.total})
        wandb.log({f"{name}_noisy_loss": self.running_loss / self.total})
        print(f"{name}_clean acc = {self.clean_running_correct / self.total}")
        print(f"{name}_noisy acc = {self.noisy_running_correct / self.total}")


class CallbackPermutationStats(Callback):
    def __init__(self):
        pass

    def on_step_end(self, metrics, name):
        pass

    def on_epoch_end(self, metrics, epoch, name):
        _, alpha_label = torch.max(metrics["alpha_matrix"].detach().cpu(), 1)
        permuted_sample = alpha_label != metrics["all_noisy_labels"]
        noisy_sample = metrics["all_noisy_labels"] != metrics["all_clean_labels"]

        num_permuted_samples = permuted_sample.sum().item()

        correct_to_false = (
            (~noisy_sample) * (alpha_label != metrics["all_clean_labels"])
        ).sum()
        false_to_correct = (
            noisy_sample * (alpha_label == metrics["all_clean_labels"])
        ).sum()
        false_to_false = (
            noisy_sample
            * (alpha_label != metrics["all_clean_labels"])
            * permuted_sample
        ).sum()

        expected_new_label_accuracy = (
            alpha_label == metrics["all_clean_labels"]
        ).sum() / metrics["all_clean_labels"].shape[0]

        # TODO: implement log_stats
        print(f"number of permuted samples = {num_permuted_samples}")
        print(f"number of correct_to_false = {correct_to_false}")
        print(f"number of false_to_correct = {false_to_correct}")
        print(f"number of false_to_false = {false_to_false}")
        print(f"expected new label accuracy = {expected_new_label_accuracy}")

        wandb.log({"num_permuted_samples": num_permuted_samples})
        wandb.log({"correct_to_false": correct_to_false})
        wandb.log({"false_to_correct": false_to_correct})
        wandb.log({"false_to_false": false_to_false})
        wandb.log({"expected_new_label_accuracy": expected_new_label_accuracy})


class CallbackLearningRateScheduler(Callback):
    def __init__(self, optimizer, max_lr, epochs, steps_per_epoch):
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch
        )

    def on_step_end(self, metrics, name):
        if name == "train":
            self.sched.step()

    def on_epoch_end(self, metrics, epoch, name):
        pass
