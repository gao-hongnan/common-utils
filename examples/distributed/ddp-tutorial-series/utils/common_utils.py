import functools
import logging
import socket
import warnings
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Union
import torch.distributed as dist
from rich.logging import RichHandler
from tabulate import tabulate

from config.base import DistributedInfo


def get_dist_info(rank: int, world_size: int) -> Dict[str, Union[int, str, bool]]:
    """Gather information about the distributed environment.

    Returns
    -------
    A dictionary containing distributed environment information.
    """
    info_dict = {
        "Explicit Rank": rank,
        "Explicit World Size": world_size,
        "Machine Hostname": socket.gethostname(),
        "PyTorch Distributed Available": dist.is_available(),
        "World Size in Initialized Process Group": dist.get_world_size(),
        "Rank within Default Process Group": dist.get_rank(),
    }

    return info_dict


def display_dist_info(
    dist_info: DistributedInfo,
    format: str = "string",  # pylint: disable=redefined-builtin
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Display information about the distributed environment in a given format or log it.

    Parameters
    ----------
    rank (int): Rank of the process.
    world_size (int): Total number of processes.
    format (str): Output format - "string" or "table".
    logger (Optional[logging.Logger]): Logger to log the information.

    Returns
    -------
    Formatted string or table if logger is not provided, otherwise logs the
    information.
    """
    dist_info: Dict[str, Union[int, str, bool]] = asdict(dist_info)

    if logger:
        for key, value in dist_info.items():
            logger.info(f"{key}: {value}")

    if format == "string":
        info_str = "\n".join([f"{key}: {value}" for key, value in dist_info.items()])
        return info_str

    elif format == "table":
        table_str = tabulate([dist_info], headers="keys", tablefmt="grid")
        if logger:
            logger.info("\n" + table_str)  # log the entire table as one message
        return table_str
    else:
        print("Invalid format specified. Choose 'string' or 'table'.")
        return


def configure_logger(rank: str, print_to_console: bool = False) -> logging.Logger:
    """
    Configure and return a logger for a given process rank.

    Parameters
    ----------
    rank : str
        The rank of the process for which the logger is being configured.

    Returns
    -------
    logging.Logger
        Configured logger for the specified process rank.
    Notes
    -----
    The logger is configured to write logs to a file named `process_{rank}.log` and
    display logs with severity level INFO and above. The reason to write each rank's
    logs to a separate file is to avoid the non-deterministic ordering of log
    messages from different ranks in the same file.
    """
    logger = logging.getLogger(f"Process-{rank}")

    # Clear existing handlers
    logger.handlers = []

    # Set the logging level
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(filename=f"process_{rank}.log")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s]: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(file_handler)

    # Console handler if needed
    if print_to_console:
        console_handler = RichHandler()
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s]: %(message)s", "%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(console_handler)

    return logger


def calculate_global_rank(
    local_rank: int, node_rank: int, num_gpus_per_node: int
) -> int:
    """Calculate the global rank of a process.

    Parameters
    ----------
    local_rank : int
        The rank of the process on the current node.
    node_rank : int
        The rank of the node on which the process is running.
    num_gpus_per_node : int
        The number of GPUs available on the current node.

    Returns
    -------
    int
        The global rank of the process.
    """
    return local_rank + node_rank * num_gpus_per_node


def deprecated(reason: str = "") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    A decorator to mark functions or methods as deprecated with a given reason.

    Args:
        reason: A string indicating why this function/method is deprecated.

    Returns:
        The decorated function/method.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"'{func.__name__}' is deprecated. {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
