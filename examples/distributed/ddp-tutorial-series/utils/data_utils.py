from torch.utils.data import Dataset, DataLoader

import torch
from typing import Tuple
from config.base import DataLoaderConfig, DistributedSamplerConfig
from dataclasses import asdict


class ToyDataset(Dataset):
    """Dummy dataset."""

    def __init__(
        self, num_samples: int, num_dimensions: int = 20, target_dimensions: int = 1
    ) -> None:
        self.num_samples = num_samples
        self.num_dimensions = num_dimensions
        self.target_dimensions = target_dimensions

        self.X = torch.randn(num_samples, num_dimensions)
        self.y = torch.randint(0, 2, (num_samples, target_dimensions)).squeeze()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.X[index]
        y = self.y[index]
        return X, y


def prepare_dataloader(dataset: Dataset, cfg: DataLoaderConfig) -> DataLoader:
    return DataLoader(dataset, **asdict(cfg))
