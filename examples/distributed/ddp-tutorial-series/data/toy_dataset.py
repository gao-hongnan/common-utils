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


def prepare_dataloader(dataset: Dataset, config: DataLoaderConfig) -> DataLoader:
    """Prepare dataloader from dataset and config."""
    return DataLoader(dataset, **asdict(config))
