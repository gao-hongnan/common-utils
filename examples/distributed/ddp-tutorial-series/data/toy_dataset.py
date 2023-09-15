from dataclasses import asdict
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from config.base import DataLoaderConfig


class ToyDataset(Dataset):
    """Dummy dataset."""

    def __init__(
        self, num_samples: int, num_dimensions: int = 20, target_dimensions: int = 1
    ) -> None:
        self.num_samples = num_samples
        self.num_dimensions = num_dimensions
        self.target_dimensions = target_dimensions

        self.X = torch.randn(num_samples, num_dimensions)
        self.y = torch.rand(num_samples, target_dimensions)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.X[index]
        y = self.y[index]
        return X, y


class ToyRegresionDataset(Dataset):
    """Dataset for regression data."""

    def __init__(
        self,
        num_samples: int,
        num_dimensions: int = 20,
        target_dimensions: int = 1,
        noise_std: float = 0.1,
    ) -> None:
        self.num_samples = num_samples
        self.num_dimensions = num_dimensions
        self.target_dimensions = target_dimensions
        self.noise_std = noise_std

        self.X, self.y = self.generate_regression_data()

    def generate_regression_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate regression data based on a linear function with added Gaussian noise."""
        # Random weights and biases to simulate a linear regression equation
        weights = torch.randn(self.num_dimensions, self.target_dimensions)
        biases = torch.randn(self.target_dimensions)

        # Generate features and targets
        X = torch.randn(self.num_samples, self.num_dimensions)
        y = X @ weights + biases

        # Add Gaussian noise
        y += self.noise_std * torch.randn_like(y)

        return X, y

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.X[index]
        y = self.y[index]
        return X, y


class ToyClassificationDataset(Dataset):
    """Dataset for classification data."""

    def __init__(
        self,
        num_samples: int,
        num_dimensions: int = 20,
        num_classes: int = 2,
        noise_prob: float = 0.1,
    ) -> None:
        self.num_samples = num_samples
        self.num_dimensions = num_dimensions
        self.num_classes = num_classes
        self.noise_prob = noise_prob

        self.X, self.y = self.generate_classification_data()

    def generate_classification_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate classification data based on a linear function with added Gaussian noise."""
        # Random weights and biases to simulate a linear classifier
        weights = torch.randn(self.num_dimensions, self.num_classes)
        biases = torch.randn(self.num_classes)

        # Generate features
        X = torch.randn(self.num_samples, self.num_dimensions)

        # Compute logits: the raw scores for each class
        logits = X @ weights + biases

        # Convert logits to probabilities using the softmax function
        probs = torch.nn.functional.softmax(logits, dim=1)

        # Generate labels based on the probabilities
        y = torch.argmax(probs, dim=1)

        # Add noise: randomly shuffle some labels
        mask = torch.rand(self.num_samples) < self.noise_prob
        y[mask] = torch.randint(0, self.num_classes, (torch.sum(mask),))

        return X, y

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.X[index]
        y = self.y[index]
        return X, y


def prepare_dataloader(dataset: Dataset, config: DataLoaderConfig) -> DataLoader:
    """Prepare dataloader from dataset and config."""
    return DataLoader(dataset, **asdict(config))
