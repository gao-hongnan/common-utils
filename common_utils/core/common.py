"""
common_utils/core/common.py

This module contains common utility functions for various purposes.
"""
import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
import yaml

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
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, **kwargs)

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
        with open(filepath, "r") as f:
            data = json.load_to_dict(f, **kwargs)
        return data


class YamlAdapter(DictPersistence):
    def save_as_dict(
        self, data: Dict[str, Any], filepath: str, **kwargs: Dict[str, Any]
    ) -> None:
        with open(filepath, "w") as f:
            yaml.safe_dump(data, f, **kwargs)

    def load_to_dict(self, filepath: str, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f, **kwargs)
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
    print(f"Using Seed Number {seed}")

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


def list_files_recursively(start_path: Union[str, Path]) -> None:
    """
    List all files and directories recursively in the given path using markdown
    style.

    Parameters
    ----------
    start_path : Union[str, Path]
        The path where the function should start listing the files and
        directories.

    Returns
    -------
    None
    """

    start_path = Path(start_path)

    def _list_files(path: Path, level: int, is_last: bool) -> None:
        """
        Helper function to list files and directories at the given path.

        Parameters
        ----------
        path : Path
            The path to list files and directories from.
        level : int
            The current depth in the file hierarchy.
        is_last : bool
            Indicates whether the current path is the last item in its parent
            directory.

        Returns
        -------
        None
        """
        prefix = (
            "    " * (level - 1) + ("└── " if is_last else "├── ") if level > 0 else ""
        )
        print(f"{prefix}{path.name}/")
        children = sorted(list(path.iterdir()), key=lambda x: x.name)
        for i, child in enumerate(children):
            if child.is_file():
                child_prefix = "    " * level + (
                    "└── " if i == len(children) - 1 else "├── "
                )
                print(f"{child_prefix}{child.name}")
            elif child.is_dir():
                _list_files(child, level + 1, i == len(children) - 1)

    _list_files(start_path, 0, False)
