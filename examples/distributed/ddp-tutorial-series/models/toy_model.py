import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ExtendedToyModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int) -> None:
        super().__init__()

        # Create a list to hold layers
        layers = []

        # Add the initial layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        # Add intermediate layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())

        # Add the final layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # Combine layers into a sequential module
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
