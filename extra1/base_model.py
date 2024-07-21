"""
base_model.py - Base model module.

Contains the base model class.

@Author: Ofir Paz
@Version: 17.07.2024
"""

# ================================== Imports ================================= #
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics import Accuracy
from torch import Tensor
from typing import Literal, Tuple, Union
# ============================== End Of Imports ============================== #


# ============================== BaseModel Class ============================= #
class BaseModel(nn.Module):
    """Base model class."""

    def __init__(self, task_type: Literal["classification", "regression"] = "classification") -> None:
        """Constructor."""
        super(BaseModel, self).__init__()
        self.best_weights: Union[dict[str, Tensor], None] = None
        self.global_epoch: int = 0
        
        self.train_costs: list[float] = []
        self.val_costs: list[float] = []
        self.train_scores: list[float] = []
        self.val_scores: list[float] = []

        if task_type not in ["classification", "regression"]:
            raise NotImplementedError("Invalid task type.")
        self.task_type: Literal["classification", "regression"] = task_type
        
        self.criterion: nn.Module = nn.CrossEntropyLoss() \
            if self.task_type == "classification" else nn.MSELoss()
        self.score_name: str = "Accuracy" if self.task_type == "classification" else "MSE"        

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

        # Create the optimizer.
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        start_epoch = self.global_epoch
        for epoch in range(num_epochs):
            running_loss = 0.
            train_score = 0.
            self.global_epoch += 1
            for mb, (x, y) in enumerate(train_loader):
                if use_cuda:
                    x: Tensor = x.cuda(); y: Tensor = y.cuda()

                y_hat, lloss = self.__train_step(x, y, optimizer, use_cuda)

                running_loss += lloss * x.size(0)

                if self.task_type == "classification":
                    train_score += (y_hat.argmax(1) == y).sum().item()

                elif self.task_type == "regression":
                    reshaped_y = y.reshape_as(y_hat)
                    train_score += F.mse_loss(y_hat, reshaped_y).item() * x.size(0)

                # TODO: Use tqdm instead of print.
                if verbose:
                    print(f"\r[epoch: {self.global_epoch:02d}/{start_epoch + num_epochs:02d} "
                          f"mb: {mb + 1:03d}/{len(train_loader):03d}]  " 
                          f"[Train loss: {lloss:.6f}]", end="")

            train_epoch_loss = running_loss / len(train_loader.dataset)  # type: ignore
            train_total_score = train_score / len(train_loader.dataset)  # type: ignore
            val_epoch_loss, val_total_score = self.calc_metrics(val_loader, use_cuda)
            
            self.train_costs.append(train_epoch_loss); self.val_costs.append(val_epoch_loss)
            self.train_scores.append(train_total_score); self.val_scores.append(val_total_score)

            # We want to maximize accuracy but minimize MSE.
            score = val_total_score if self.task_type == "classification" else -val_total_score
            self.save_best_weights(score)

            if verbose and (epoch % print_stride == 0 or epoch == num_epochs - 1):
                print(f"\r[epoch: {self.global_epoch:02d}/{start_epoch + num_epochs:02d}] "
                      f"[Train loss: {train_epoch_loss:.6f} "
                      f" Train {self.score_name}: {train_total_score:.3f}]  "
                      f"[Val loss: {val_epoch_loss:.6f}] "
                      f" Val {self.score_name}: {val_total_score:.3f}]")
        self.cpu()

    def __train_step(self, x: Tensor, y: Tensor, optimizer: optim.Optimizer,
                     use_cuda: bool) -> Tuple[Tensor, float]:
        """
        Performs a single training step.

        Args:
            x (Tensor) - Input tensor.
            y (Tensor) - Target tensor.
            optimizer (Optimizer) - Optimizer.
            use_cuda (bool) - Use CUDA flag.

        """
        # zero the parameter gradients.
        optimizer.zero_grad()

        # forward + backward + optimize.
        y_hat = self(x)

        if self.task_type == "regression":
            y = y.reshape_as(y_hat)

        loss = self.criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        # Calc loss.
        lloss = loss.item()

        return y_hat, lloss

    def calc_metrics(self, data_loader: DataLoader, use_cuda: bool) -> Tuple[float, float]:
        """
        Calculates and returns the loss and the score of the model on a give dataset.
        the score is the accuracy for classification tasks and the MSE for regression tasks.

        Args:
            data_loader (DataLoader) - Data loader.
            use_cuda (bool) - Use CUDA flag.

        Returns:
            Tuple[float, float] - The loss and score of the model on the given dataset.            
        """
        running_loss = 0.
        score = 0.
        self.eval()
        with torch.no_grad():
            for x, y in data_loader:
                if use_cuda:
                    x: Tensor = x.cuda(); y: Tensor = y.cuda()
                
                y_hat = self(x)

                if self.task_type == "classification":
                    score += (torch.argmax(y_hat, dim=1) == y).sum().item()

                elif self.task_type == "regression":
                    y = y.reshape_as(y_hat)
                    score += F.mse_loss(y_hat, y).item() * x.size(0)

                running_loss += self.criterion(y_hat, y).item() * x.size(0)

        self.train()
        total_loss = running_loss / len(data_loader.dataset)  # type: ignore
        total_score = score / len(data_loader.dataset)  # type: ignore
        return total_loss, total_score

    def save_best_weights(self, score: float) -> None:
        """Saves the best weights of the model."""
        if self.best_weights is None:
            self.best_weights = copy.deepcopy(self.state_dict())
        elif score > max(self.val_scores):
            self.best_weights = copy.deepcopy(self.state_dict())

    def load_best_weights(self):
        """Loads the best weights of the model."""
        assert self.best_weights is not None, "No weights to load."
        self.load_state_dict(self.best_weights)

    def get_outputs(self, data_loader: DataLoader, try_cuda: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Calculates and returns the outputs of the model on a given dataset.
        
        **Note: Useless for regression tasks.**

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
