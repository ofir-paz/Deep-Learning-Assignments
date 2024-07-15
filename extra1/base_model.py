"""
base_model.py - Base model module.

Contains the base model class.

@Author: Ofir Paz
@Version: 18.06.2024
"""

# ================================== Imports ================================= #
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics import Accuracy
from torch import Tensor
from typing import Tuple, Union
# ============================== End Of Imports ============================== #


# ============================== BaseModel Class ============================= #
class BaseModel(nn.Module):
    """Base model class."""

    def __init__(self) -> None:
        """Constructor."""
        super(BaseModel, self).__init__()
        self.best_weights: Union[dict[str, Tensor], None] = None
        self.global_epoch: int = 0
        self.costs: list[float] = []
        self.train_accs: list[float] = []
        self.val_accs: list[float] = []

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        raise NotImplementedError

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            num_epochs: int = 30, lr: float = 0.001, wd: float = 0.,
            try_cuda: bool = True, verbose: bool = True, print_stride: int = 1) -> None:
        """
        Base function for training a model.

        Args:
            train_loader (DataLoader) - The dataloader to fit the model to.
            val_loader (DataLoader) - The dataloader to validate the model on.
            num_epochs (int) - Number of epochs.
            lr (float) - Learning rate.
            wd (float) - Weight decay.
            try_cuda (bool) - Try to use CUDA.
            verbose (bool) - Verbose flag.
            print_stride (int) - Print stride (in epochs).
        """
        use_cuda = try_cuda and torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            print("Using CUDA for training.")
        else:
            self.cpu()
            print("Using CPU for training.")

        # Create the optimizer and criterion.
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        start_epoch = self.global_epoch
        for epoch in range(num_epochs):
            train_true = 0.
            running_loss = 0.
            self.global_epoch += 1
            for mb, (x, y) in enumerate(train_loader):
                x: Tensor; y: Tensor  # Fix type hinting.

                y_hat, lloss = self.__train_step(x, y, optimizer, criterion, use_cuda)

                running_loss += lloss * x.size(0)

                # Calc accuracy.
                train_true += (y_hat.argmax(1) == y).sum().item()

                # TODO: Use tqdm instead of print.
                if verbose:
                    print(f"\r[epoch: {self.global_epoch:02d}/{start_epoch + num_epochs:02d}", end=" ")
                    print(f"mb: {mb + 1:03d}/{len(train_loader):03d}]", end=" ")
                    print(f"loss: {lloss:.6f}", end="")

            epoch_loss = running_loss / len(train_loader.dataset)  # type: ignore
            train_acc = train_true / len(train_loader.dataset)  # type: ignore
            val_acc = self.calc_acc(val_loader, use_cuda)
            self.save_best_weights(val_acc)
            self.costs.append(epoch_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            if verbose and (epoch % print_stride == 0 or epoch == num_epochs - 1):
                print(f"\r[epoch: {self.global_epoch:02d}/{start_epoch + num_epochs:02d}]", end=" ")
                print(f"[Total Loss: {epoch_loss:.6f}]", end=" ")
                print(f"[Train Acc: {100 * train_acc:.3f}%]", end=" ")
                print(f"[Val Acc: {100 * val_acc:.3f}%]")

    def __train_step(self, x: Tensor, y: Tensor, optimizer: optim.Optimizer,
                     criterion: nn.Module, use_cuda: bool) -> Tuple[Tensor, float]:
        """
        Performs a single training step.

        Args:
            x (Tensor) - Input tensor.
            y (Tensor) - Target tensor.
            optimizer (Optimizer) - Optimizer.
            criterion (Loss) - Loss function.
            use_cuda (bool) - Use CUDA flag.

        """
        if use_cuda:
            x, y = x.cuda(), y.cuda()

        # zero the parameter gradients.
        optimizer.zero_grad()

        # forward + backward + optimize.
        y_hat = self(x)

        loss = criterion(y_hat, y)
        loss.backward(retain_graph=True)
        optimizer.step()

        # Calc loss.
        lloss = loss.item()

        return y_hat, lloss

    def calc_acc(self, data_loader: DataLoader, use_cuda: bool) -> float:
        """
        Calculates and returns the accuracy of the model on a give dataset.

        Args:
            data_loader (DataLoader) - Data loader.
            use_cuda (bool) - Use CUDA flag.

        Returns:
            float - The accuracy of the model on the given dataset.
        """
        accuracy = Accuracy(task="multiclass", num_classes=10)
        if use_cuda:
            accuracy = accuracy.cuda()
        self.eval()
        acc = 0.
        with torch.no_grad():
            for x, y in data_loader:
                if use_cuda:
                    x, y = x.cuda(), y.cuda()
                y_hat = self(x)
                acc += accuracy(y_hat, y).item() * x.size(0)

        self.train()
        return acc / len(data_loader.dataset)  # type: ignore

    def save_best_weights(self, val_acc: float) -> None:
        """Saves the best weights of the model."""
        if self.best_weights is None:
            self.best_weights = copy.deepcopy(self.state_dict())
        elif val_acc > max(self.val_accs):
            self.best_weights = copy.deepcopy(self.state_dict())
        self.val_accs.append(val_acc)

    def load_best_weights(self):
        """Loads the best weights of the model."""
        assert self.best_weights is not None, "No weights to load."

        self.load_state_dict(self.best_weights)

    def get_outputs(self, data_loader: DataLoader, try_cuda: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Calculates and returns the outputs of the model on a given dataset.

        Args:
            data_loader (DataLoader) - Data loader.
            try_cuda (bool) - Try to use CUDA flag.

        Returns:
            Tuple[Tensor, Tensor] - The outputs (class, probability) of the model on the given dataset.
        """
        use_cuda = try_cuda and torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        self.eval()
        outputs: list[Tensor] = []
        with torch.no_grad():
            for x, _ in data_loader:
                if use_cuda:
                    x = x.cuda()
                y_hat = self(x)
                outputs.append(y_hat)
        self.train()
        logits, preds = torch.cat(outputs).max()
        probs = torch.softmax(logits, dim=1)

        return preds, probs

# ========================== End Of BaseModel Class ========================== #
