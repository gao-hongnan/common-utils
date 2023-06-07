"""
common_utils/core/common.py

This module contains common utility functions for various purposes.
"""
import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import rich
import torch
import yaml
from yaml import FullLoader

from common_utils.core.base import DictPersistence


class JsonAdapter(DictPersistence):
    def save_as_dict(
        self, data: Dict[str, Any], filepath: str, **kwargs: Dict[str, Any]
    ) -> None:
        """
        Save a dictionary to a specific location.

        Parameters
        ----------
        data : Dict[str, Any]
            Data to save.
        filepath : str
            Location of where to save the data.
        cls : Type, optional
            Encoder to use on dict data, by default None.
        sortkeys : bool, optional
            Whether to sort keys alphabetically, by default False.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, **kwargs)

    def load_to_dict(self, filepath: str, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Load a dictionary from a JSON's filepath.

        Parameters
        ----------
        filepath : str
            Location of the JSON file.

        Returns
        -------
        data: Dict[str, Any]
            Dictionary loaded from the JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f, **kwargs)
        return data


class YamlAdapter(DictPersistence):
    def save_as_dict(
        self, data: Dict[str, Any], filepath: str, **kwargs: Dict[str, Any]
    ) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, **kwargs)

    def load_to_dict(self, filepath: str, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=FullLoader, **kwargs)
        return data


def seed_all(seed: Optional[int] = 1992, seed_torch: bool = True) -> None:
    """
    Seed all random number generators.

    Parameters
    ----------
    seed : int, optional
        Seed number to be used, by default 1992.
    seed_torch : bool, optional
        Whether to seed PyTorch or not, by default True.
    """
    rich.print(f"Using Seed Number {seed}")

    # fmt: off
    os.environ["PYTHONHASHSEED"] = str(seed)        # set PYTHONHASHSEED env var at fixed value
    np.random.seed(seed)                            # numpy pseudo-random generator
    random.seed(seed)                               # python's built-in pseudo-random generator

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)                # pytorch (both CPU and CUDA)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    # fmt: on


def seed_worker(_worker_id: int, seed_torch: bool = True) -> None:
    """
    Seed a worker with the given ID.

    Parameters
    ----------
    _worker_id : int
        Worker ID to be used for seeding.
    seed_torch : bool, optional
        Whether to seed PyTorch or not, by default True.

    """
    worker_seed = (
        torch.initial_seed() % 2**32 if seed_torch else random.randint(0, 2**32 - 1)
    )
    np.random.seed(worker_seed)
    random.seed(worker_seed)
