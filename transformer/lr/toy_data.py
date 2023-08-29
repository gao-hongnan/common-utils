import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import torch
from typing import Tuple


def generate_toy_data(seed: int, n_data: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate toy data for binary classification.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_data : int
        Number of data points.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Features and labels as PyTorch tensors.
    """
    np.random.seed(seed)
    x_data = np.random.normal(0, 1, size=(n_data, 2))
    y_data = (np.sum(x_data, axis=1) > 0).astype(int)
    print(y_data)

    x_data_torch = torch.tensor(x_data, dtype=torch.float32)
    y_data_torch = torch.tensor(y_data, dtype=torch.long)

    return x_data_torch, y_data_torch

# Generate the toy data
x_data_torch, y_data_torch = generate_toy_data(seed=0, n_data=100)
labels = np.unique(y_data_torch.numpy())
# Plot the generated toy data
plt.figure(figsize=(8, 8))

# Scatter plot for Class 0
# need colors for each class
plt.scatter(x_data_torch[y_data_torch == 0][:, 0], x_data_torch[y_data_torch == 0][:, 1],
            label='Class 0', edgecolors='k',  facecolors='g', marker='o')

# Scatter plot for Class 1
plt.scatter(x_data_torch[y_data_torch == 1][:, 0], x_data_torch[y_data_torch == 1][:, 1],
            label='Class 1', edgecolors='k', facecolors='b', marker='x')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Toy Data')
plt.legend()
plt.colorbar(label='Label')
plt.grid(True)
plt.show()

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

class SimpleModel(nn.Module):
    """A simple linear model for binary classification."""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)


def train_model(model: nn.Module,
                optimizer: optim.Optimizer,
                scheduler: CosineAnnealingLR,
                x_data: torch.Tensor,
                y_data: torch.Tensor,
                n_epochs: int) -> Tuple[nn.Module, np.ndarray]:
    """
    Train a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to train.
        optimizer (optim.Optimizer): The optimizer.
        scheduler (CosineAnnealingLR): The learning rate scheduler.
        x_data (torch.Tensor): Feature tensor.
        y_data (torch.Tensor): Label tensor.
        n_epochs (int): Number of epochs to train.

    Returns:
        model (nn.Module): The trained model.
        losses (np.ndarray): Array of loss values.
    """
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(x_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    return model, np.array(losses)


# Initialize the model, optimizer, and scheduler
model = SimpleModel()
optimizer = optim.AdamW(model.parameters(), lr=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.01)

# Train the model
trained_model, losses = train_model(model, optimizer, scheduler, x_data_torch, y_data_torch, n_epochs=100)

# Plot the loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

from matplotlib import cm

def plot_loss_contour(model: nn.Module, x_data: torch.Tensor, y_data: torch.Tensor,
                      xlim: Tuple[float, float], ylim: Tuple[float, float], n_points: int = 100):
    """
    Plot the loss contour for the model given the data.

    Args:
        model (nn.Module): The trained model.
        x_data (torch.Tensor): Feature tensor.
        y_data (torch.Tensor): Label tensor.
        xlim (Tuple[float, float]): Limits for the x-axis.
        ylim (Tuple[float, float]): Limits for the y-axis.
        n_points (int): Number of points for the contour plot.
    """
    criterion = nn.CrossEntropyLoss()

    x_vals = np.linspace(xlim[0], xlim[1], n_points)
    y_vals = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            model.fc.weight.data[0][0] = X[i, j]
            model.fc.weight.data[0][1] = Y[i, j]
            output = model(x_data)
            loss = criterion(output, y_data)
            Z[i, j] = loss.item()

    plt.contourf(X, Y, Z, levels=50, cmap=cm.coolwarm)
    plt.colorbar()
    plt.xlabel('Weight 1')
    plt.ylabel('Weight 2')
    plt.title('Loss Contour')
    plt.show()

# Define limits for the weights
xlim = (-2, 2)
ylim = (-2, 2)

# Generate the contour plot
plot_loss_contour(trained_model, x_data_torch, y_data_torch, xlim, ylim)
