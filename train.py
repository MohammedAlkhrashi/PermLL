import torch
from torch.nn import nn


class TrainPermutation:
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        val_loader,
        epochs,
        criterion,
        gpu_num="0",
    ) -> None:
        self.device = torch.device(
            f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
        )
        self.model: nn.Module = model.to(self.device)
        self.optimizer: torch.optim.Optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs

        self.train_loader = train_loader
        self.val_loader = val_loader

    def start(self):
        for epoch in range(self.epochs):

            self.model.train()
            for (image, label) in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.criterion(output, label)

