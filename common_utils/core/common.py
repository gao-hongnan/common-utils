"""
common_utils/core/common.py

This module contains common utility functions for various purposes.
"""
import json
import os
import random
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import rich
import torch
import yaml
from dotenv import load_dotenv
from yaml import FullLoader

from common_utils.core.base import DictPersistence
from common_utils.core.logger import Logger

# Setup logging
# logging.basicConfig(
#     level="INFO",
#     format="%(asctime)s [%(levelname)s]: %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     handlers=[RichHandler()],
# )

# logger = logging.getLogger("rich")


# Setup logging
logger = Logger(
    module_name=__name__, propagate=False, log_root_dir=None, log_file=None
).logger


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


def generate_uuid() -> str:
    """Generate a UUID.

    Returns
    -------
    str
        UUID1 as a string.
    """
    return str(uuid.uuid1())


def seed_all(seed: Optional[int] = 1992, seed_torch: bool = True) -> int:
    """
    Seed all random number generators.

    Parameters
    ----------
    seed : int, optional
        Seed number to be used, by default 1992.
    seed_torch : bool, optional
        Whether to seed PyTorch or not, by default True.
    """
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
    return seed


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


def load_env_vars(root_dir: Union[str, Path]) -> Dict[str, str]:
    """
    Load environment variables from .env.default and .env files.

    Parameters
    ----------
    root_dir: Union[str, Path]
        Root directory of the .env files.

    Returns
    -------
    Dict[str, str]
        Dictionary with the environment variables.
    """

    if isinstance(root_dir, str):
        root_dir = Path(root_dir)

    load_dotenv(dotenv_path=root_dir / ".env.default")
    load_dotenv(dotenv_path=root_dir / ".env", override=True)

    return dict(os.environ)


def get_root_dir(env_var: str = "ROOT_DIR", root_dir: str = ".") -> Path:
    """
    Get the root directory of the project.

    Parameters
    ----------
    env_var: str
        Name of the environment variable to use. Defaults to "ROOT_DIR".
    root_dir: str
        Default value to use if the environment variable is not set. Defaults to ".".

    Returns
    -------
    Path
        Path to the root directory of the project.
    """
    root_dir_env = os.getenv(env_var)
    if root_dir_env is None:
        logger.warning("Environment variable %s is not set.", env_var)
        logger.warning("Using default value %s", root_dir)
        return Path(root_dir)
    logger.info(
        "Using environment variable %s and discarding default value %s",
        env_var,
        root_dir,
    )
    return Path(root_dir_env)
