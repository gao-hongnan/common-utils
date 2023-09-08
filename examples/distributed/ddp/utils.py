import logging
import torch.distributed as dist
import socket
from typing import Dict, Union, Optional
from tabulate import tabulate


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
    rank: int,
    world_size: int,
    format: str = "string", # pylint: disable=redefined-builtin
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
    info_dict = get_dist_info(rank, world_size)

    if logger:
        for key, value in info_dict.items():
            logger.info(f"{key}: {value}")

    if format == "string":
        info_str = "\n".join([f"{key}: {value}" for key, value in info_dict.items()])
        return info_str

    elif format == "table":
        return tabulate([info_dict], headers="keys", tablefmt="grid")

    else:
        print("Invalid format specified. Choose 'string' or 'table'.")
        return


def configure_logger(rank: int) -> logging.Logger:
    """
    Configure and return a logger for a given process rank.

    Parameters
    ----------
    rank : int
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
    handlers = [logging.FileHandler(filename=f"process_{rank}.log")]  # , RichHandler()]
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(f"Process-{rank}")
