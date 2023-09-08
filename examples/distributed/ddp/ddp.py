"""
qsub -I -l select=1:ngpus=4 -P 11003281 -l walltime=24:00:00 -q ai
"""
import logging
import os
import socket
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rich.logging import RichHandler
from multigpu import prepare_dataloader, load_train_objs, Trainer
from transformers import (
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    AutoModel,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class InitEnvArgs:
    """Initialize environment variables. The attribute must be
    named such that the upper case version is the same as the
    environment variable name.
    """

    master_addr: str = field(
        default="localhost", metadata={"help": "IP address of the machine"}
    )
    master_port: str = field(default="12356", metadata={"help": "The port number"})


def init_env(cfg: InitEnvArgs) -> None:
    """Initialize environment variables."""
    cfg: Dict[str, str] = asdict(cfg)

    for key, value in cfg.items():
        upper_key = key.upper()
        os.environ[upper_key] = value


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class InitProcessGroupArgs:
    """From torch/distributed/distributed_c10d.py:
    There are 2 main ways to initialize a process group:
    1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
    2. Specify ``init_method`` (a URL string) which indicates where/how
        to discover peers. Optionally specify ``rank`` and ``world_size``,
        or encode all required parameters in the URL and omit them.

    If neither is specified, ``init_method`` is assumed to be "env://".
    """

    rank: int = field(default=-1, metadata={"help": "Rank of the current process"})
    world_size: int = field(
        default=-1, metadata={"help": "Number of processes participating in the job"}
    )
    backend: str = field(
        default="nccl", metadata={"help": "Name of the backend to use"}
    )
    init_method: str = field(
        default="env://",
        metadata={"help": "URL specifying how to initialize the package"},
    )


def init_process(
    cfg: InitProcessGroupArgs,
    logger: logging.Logger,
    func: Optional[Callable] = None,
) -> None:
    """Initialize the distributed environment via init_process_group."""

    logger = configure_logger(rank=cfg.rank)

    dist.init_process_group(**asdict(cfg))
    logger.info(f"Initialized process group: Rank {cfg.rank} out of {cfg.world_size}")

    display_dist_info(cfg.rank, cfg.world_size, logger)

    if func is not None:
        func(cfg.rank, cfg.world_size)


def display_dist_info(rank: int, world_size: int, logger: logging.Logger) -> None:
    logger.info(f"Explicit Rank: {rank}")
    logger.info(f"Explicit World Size: {world_size}")
    logger.info(f"Machine Hostname: {socket.gethostname()}")
    logger.info(f"PyTorch Distributed Available: {dist.is_available()}")
    logger.info(f"World Size in Initialized Process Group: {dist.get_world_size()}")

    group_rank = dist.get_rank()
    logger.info(f"Rank within Default Process Group: {group_rank}")


def run(rank: int, world_size: int) -> None:
    """Blocking point-to-point communication."""
    tensor = torch.zeros(1)
    print("Rank ", rank, " has data ", tensor[0])
    tensor = tensor.to(rank)  # in 1 node, global rank = local rank
    if rank == 0:
        tensor += 1
        # Send the tensor to processes other than 0
        for other_rank in range(1, world_size):  # Sending to all other ranks
            dist.send(tensor=tensor, dst=other_rank)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)

    print("Rank ", rank, " has data ", tensor[0])


def main(world_size: int) -> None:
    """Main driver function."""
    init_env(cfg=InitEnvArgs())
    processes = []
    mp.set_start_method("spawn")
    logger = configure_logger("main")

    for rank in range(world_size):
        init_process_group_args = InitProcessGroupArgs(rank=rank, world_size=world_size)
        p = mp.Process(
            target=init_process,
            args=(init_process_group_args, logger, run),
            # kwargs={},
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    # parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    # parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument(
        "--world_size", default=None, type=int, help="Total number of GPUs"
    )

    args = parser.parse_args()

    if not args.world_size:
        world_size = torch.cuda.device_count()
    else:
        world_size = args.world_size

    main(world_size)
