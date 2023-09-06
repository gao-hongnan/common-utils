"""
qsub -I -l select=1:ngpus=4 -P 11003281 -l walltime=24:00:00 -q ai
"""
import logging
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rich.logging import RichHandler
from multigpu import prepare_dataloader, load_train_objs, Trainer


def configure_logger(rank: int) -> logging.Logger:
    handlers = [logging.FileHandler(filename=f"process_{rank}.log")]  # , RichHandler()]
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(f"Process-{rank}")


def init_env(**kwargs: Dict[str, Any]) -> None:
    """Initialize environment variables."""
    os.environ["MASTER_ADDR"] = kwargs.pop(
        "master_addr", "localhost"
    )  # IP address of the machine
    os.environ["MASTER_PORT"] = kwargs.pop("master_port", "12356")  # port number
    os.environ.update(kwargs)


def init_process(
    rank: int,
    world_size: int,
    backend: str,
    logger: logging.Logger,
    fn: Optional[Callable] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """Initialize the distributed environment via init_process_group."""

    logger = configure_logger(rank)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, **kwargs)
    logger.info(f"Initialized process group: Rank {rank} out of {world_size}")

    display_dist_info(rank, world_size, logger)

    if fn is not None:
        fn(rank, world_size)


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
    init_env()
    processes = []
    mp.set_start_method("spawn")
    logger = configure_logger("main")

    for rank in range(world_size):
        p = mp.Process(
            target=init_process,
            args=(rank, world_size, "nccl", logger, run),
            kwargs={"init_method": "env://"},
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
