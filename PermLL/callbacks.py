import os
from typing import List

import torch
from termcolor import colored
from torch.utils.data import DataLoader

import wandb
from group_utils import GroupPicker
from model import GroupModel


class Callback:
    def early_stop(self):
        return False

    def on_step_end(self, metrics, name):
        raise NotImplementedError

    def on_epoch_end(self, metrics, epoch, name):
        raise NotImplementedError

    def on_training_end(self, metrics):
        raise NotImplementedError


class IdentityCallback(Callback):
    def on_step_end(self, metrics, name):
        pass

    def on_epoch_end(self, metrics, epoch, name):
        pass

    def on_training_end(self, metrics):
        pass


class CallbackNoisyStatistics(Callback):
    def __init__(self, max_no_improvement=20) -> None:
        self.reset()
        self.best_clean_acc = 0
        self.test_best_clean_acc = 0
        self.count_no_improvment = 0
        self.test_count_no_improvment = 0

        self.max_no_improvement = max_no_improvement

    def reset(self):
        self.running_loss = 0
        self.noisy_running_correct = 0
        self.clean_running_correct = 0
        self.total = 0
        self.all_train_prediction_before_perm_list = []
        self.all_sample_index_list = []

    def early_stop(self):
        return self.count_no_improvment == self.max_no_improvement

    def on_step_end(self, metrics, name):
        average_output = self.average_prediction(metrics["output"])
        _, predicted = torch.max(average_output.detach(), 1)

        average_unpermuted_logits = self.average_prediction(
            metrics["unpermuted_logits"]
        )
        _, prediction_before_perm = torch.max(average_unpermuted_logits.detach(), 1)

        self.running_loss += metrics["loss"]
        self.noisy_running_correct += (
            (predicted == metrics["batch"]["noisy_label"]).sum().item()
        )
        self.clean_running_correct += (
            (prediction_before_perm == metrics["batch"]["true_label"]).sum().item()
        )
        self.total += metrics["batch"]["noisy_label"].size(0)

        if name == "train":
            self.all_train_prediction_before_perm_list.append(prediction_before_perm)
            self.all_sample_index_list.append(metrics["sample_index"])

    def on_epoch_end(self, metrics, epoch, name):
        self.log_stats(name, epoch, metrics)
        if name == "train":
            self.all_train_prediction_before_perm = (
                torch.cat(self.all_train_prediction_before_perm_list).detach().cpu()
            )
            self.all_sample_index = torch.cat(self.all_sample_index_list).detach().cpu()
        self.reset()

    def on_training_end(self, metrics):
        pass

    def log_stats(self, name, epoch, metrics):
        if self.total == 0:
            return
        clean_acc = self.clean_running_correct / self.total
        noisy_acc = self.noisy_running_correct / self.total

        if name == "val":
            metrics["val_clean_acc"] = clean_acc
            metrics["val_noisy_acc"] = noisy_acc
            if clean_acc > self.best_clean_acc:
                self.best_clean_acc = clean_acc
                self.count_no_improvment = 0
            else:
                self.count_no_improvment += 1

            wandb.log({f"val_best_clean_acc": self.best_clean_acc})

        if name == "test":
            if self.count_no_improvment == 0:  # this epoch was best val acc so update
                wandb.log({f"test_acc_on_best_val": clean_acc})
            if clean_acc > self.test_best_clean_acc:
                self.test_best_clean_acc = clean_acc
                self.test_count_no_improvment = 0
            else:
                self.test_count_no_improvment += 1

            wandb.log({f"test_best_clean_acc": self.test_best_clean_acc})

        wandb.log({f"{name}_clean_accuracy": clean_acc})
        wandb.log({f"{name}_noisy_accuracy": noisy_acc})
        wandb.log({f"{name}_noisy_loss": self.running_loss / self.total})
        print(f"{name}_clean acc = {self.clean_running_correct / self.total}")
        print(f"{name}_noisy acc = {self.noisy_running_correct / self.total}")

    def average_prediction(self, output: List[torch.tensor]):
        output = torch.stack(output)  # shape = (num_networks,batch,classes)
        output_props = torch.softmax(
            output, dim=-1
        )  # this works even if output is log_softmax. (softmax(log_softmax(x)) = softmax(x))
        avg_output_props = output_props.mean(0)  # average over networks
        return avg_output_props


def permuted_samples(metrics):
    _, alpha_label = torch.max(metrics["alpha_matrix"].detach().cpu(), 1)
    sample_permuted = alpha_label != metrics["all_noisy_labels"]
    num_permuted_samples = sample_permuted.sum().item()
    return alpha_label, sample_permuted, num_permuted_samples


class CallbackPermutationStats(Callback):
    def __init__(self):
        pass

    def on_step_end(self, metrics, name):
        pass

    def on_epoch_end(self, metrics, epoch, name):
        if name != "train":
            return
        alpha_label, sample_permuted, self.num_permuted_samples = permuted_samples(
            metrics
        )
        noisy_sample = metrics["all_noisy_labels"] != metrics["all_clean_labels"]
        self.correct_to_false = (
            (~noisy_sample)
            * (
                alpha_label != metrics["all_clean_labels"]
            )  # TODO:check this what if we relabel
        ).sum()
        self.false_to_correct = (
            noisy_sample * (alpha_label == metrics["all_clean_labels"])
        ).sum()
        self.false_to_false = (
            noisy_sample
            * (alpha_label != metrics["all_clean_labels"])
            * sample_permuted
        ).sum()
        self.expected_new_label_accuracy = (
            alpha_label == metrics["all_clean_labels"]
        ).sum() / metrics["all_clean_labels"].shape[0]

        self.log_stats(name)

        print(
            f"number of permuted samples based on alpha_matrix = {self.num_permuted_samples}: ({colored(self.false_to_correct.item(), 'green')},  {colored(self.false_to_false.item(), 'yellow')}, {colored(self.correct_to_false.item(), 'red')})"
        )
        print(f"expected new label accuracy = {self.expected_new_label_accuracy}")

    def on_training_end(self, metrics):
        pass

    def log_stats(self, name):
        wandb.log({"num_permuted_samples": self.num_permuted_samples})
        wandb.log({"correct_to_false": self.correct_to_false})
        wandb.log({"false_to_correct": self.false_to_correct})
        wandb.log({"false_to_false": self.false_to_false})
        wandb.log({"expected_new_label_accuracy": self.expected_new_label_accuracy})


class AdaptiveNetworkLRScheduler(Callback):
    def __init__(self, optimizer, init_lr, num_stat_samples, beta, factor=10):
        self.optimizer: torch.optim.Optimizer = optimizer
        self.num_stat_samples = num_stat_samples
        self.factor = factor
        self.init_lr = init_lr
        self.beta = beta
        self.memorization_metric = 1

    def on_step_end(self, metrics, name):
        if name == "train":
            all_losses = metrics["all_losses"]
            small_losses = all_losses[: self.num_stat_samples // 2]
            large_losses = all_losses[-self.num_stat_samples // 2 :]
            self.memorization_metric = (
                self.beta * self.memorization_metric
                + (1 - self.beta) * small_losses.mean() / large_losses.mean()
            )
            lr = (self.factor * self.memorization_metric).clamp(max=1) * self.init_lr

            if lr < 0.0001:
                lr = 0

            for g in self.optimizer.param_groups:
                g["lr"] = lr
            wandb.log({"learning_rate": self.optimizer.param_groups[0]["lr"]})

    def on_epoch_end(self, metrics, epoch, name):
        pass

    def on_training_end(self, metrics):
        pass


class StepLRLearningRateScheduler(Callback):
    def __init__(self, optimizer, milestones: str, gamma, last_epoch=-1):
        self.sched = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch
        )

    def on_step_end(self, metrics, name):
        pass

    def on_epoch_end(self, metrics, epoch, name):
        if name == "train":
            self.sched.step()
            wandb.log({"learning_rate": self.sched.get_lr()[0]})

    def on_training_end(self, metrics):
        pass


class OneCycleLearningRateScheduler(Callback):
    def __init__(self, optimizer, max_lr, epochs, steps_per_epoch, final_div_factor):
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            final_div_factor=final_div_factor,
        )

    def on_step_end(self, metrics, name):
        if name == "train":
            self.sched.step()

    def on_epoch_end(self, metrics, epoch, name):
        if name == "train":
            wandb.log({"learning_rate": self.sched.get_lr()[0]})

    def on_training_end(self, metrics):
        pass


class CosineAnnealingLRScheduler(Callback):
    def __init__(self, optimizer, T_0, T_mult=1):
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLRScheduler(
            optimizer, T_0=T_0, T_mult=T_mult
        )

    def on_step_end(self, metrics, name):
        if name == "train":
            self.sched.step()

    def on_epoch_end(self, metrics, epoch, name):
        if name == "train":
            wandb.log({"learning_rate": self.sched.get_lr()[0]})

    def on_training_end(self, metrics):
        pass


class AdaptivePermLRScheduler(Callback):
    def __init__(self, optimizer, min_acc_threshold, adaptive_lr_mode, max_lr=900):
        self.optimizer: torch.optim.Optimizer = optimizer
        self.max_lr = max_lr
        self.min_acc_threshold = min_acc_threshold
        self.adaptive_lr_mode = adaptive_lr_mode

    def on_step_end(self, metrics, name):
        pass

    def on_epoch_end(self, metrics, epoch, name):
        if name == "train":
            wandb.log({"perm_learning_rate": self.optimizer.param_groups[0]["lr"]})
        if name == "val":
            last_accuracy = metrics["val_clean_acc"]
            if self.adaptive_lr_mode == "linear":
                new_lr = (
                    self.max_lr
                    * (last_accuracy - self.min_acc_threshold)
                    / (1 - self.min_acc_threshold)
                )
                new_lr = max(0, new_lr)
            elif self.adaptive_lr_mode == "constant":
                new_lr = self.max_lr if last_accuracy > self.min_acc_threshold else 0
            for g in self.optimizer.param_groups:
                g["lr"] = new_lr

    def on_training_end(self, metrics):
        pass


class CallbackGroupPickerReseter(Callback):
    def __init__(self, group_picker: GroupPicker):
        self.group_picker = group_picker
        self.count = 0

    def on_step_end(self, metrics, name):
        self.count += 1
        if name == "train" and self.count % self.group_picker.num_groups == 0:
            self.group_picker.create_new_groups()

    def on_epoch_end(self, metrics, epoch, name):
        pass

    def on_training_end(self, metrics):
        pass


class CallbackLabelCorrectionStats(Callback):
    def __init__(self):
        self.main_log_dir = "stats_logs"
        os.makedirs(self.main_log_dir, exist_ok=True)

    def on_step_end(self, metrics, name):
        pass

    def on_epoch_end(self, metrics, epoch, name):
        pass

    def on_training_end(self, metrics):
        log_dir = os.path.join(self.main_log_dir, wandb.run.name)
        os.makedirs(log_dir, exist_ok=True)
        loged_metrics = ["all_noisy_labels", "all_clean_labels", "alpha_matrix"]
        for metric in loged_metrics:
            torch.save(
                metrics[metric].detach().cpu(), os.path.join(log_dir, metric + ".pt")
            )


class CallbackReweightSampler(Callback):
    def __init__(self, data_loader: DataLoader, num_classes):
        self.data_loader = data_loader
        self.num_classes = num_classes

    def on_step_end(self, metrics, name):
        pass

    def on_epoch_end(self, metrics, epoch, name):
        if name != "train":
            return

        _, corrected_labels = torch.max(metrics["alpha_matrix"].detach().cpu(), 1)
        N = len(corrected_labels)
        weights = [None] * N
        class_count = [0] * self.num_classes
        for i in range(N):
            label = corrected_labels[i]
            class_count[label.item()] += 1
        for i in range(N):
            label = corrected_labels[i]
            weights[i] = 1 / class_count[label.item()]

        new_weights = torch.DoubleTensor(weights)
        self.data_loader.sampler.weights = new_weights

    def on_training_end(self, metrics):
        pass


class CallbackWarmupPerm(Callback):
    def __init__(self, model: GroupModel, warmup):
        assert (
            not model.perm_model.disable_module
        ), "To use warmup enable do not disable perm"

        self.model = model
        self.warmup = warmup
        self.step_count = 0

        self.model.perm_model.disable_module = True

    def on_step_end(self, metrics, name):
        if name != "train":
            return

        if self.warmup == self.step_count:
            self.model.perm_model.disable_module = False
        else:
            self.step_count += 1

    def on_epoch_end(self, metrics, epoch, name):
        pass

    def on_training_end(self, metrics):
        pass
