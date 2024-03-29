"""
qsub -I -l select=1:ngpus=4 -P <project_name> -l walltime=24:00:00 -q <queue_name>
module load cuda/<cuda_version> or module load cuda for latest version
cd examples/distributed/ddp-tutorial-series && \
export PYTHONPATH=$PYTHONPATH:$(pwd) && \
python 03_writing_distributed_applications_with_pytorch/01_point_to_point.py --world_size 4
"""
import logging
import os
from dataclasses import asdict
from typing import Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from config.base import DistributedInfo, InitEnvArgs, InitProcessGroupArgs
from utils.common_utils import configure_logger, display_dist_info, init_env


def init_process(
    config: InitProcessGroupArgs,
    node_rank: int,
    logger: logging.Logger,
    func: Optional[Callable] = None,
) -> None:
    """Initialize the distributed environment via init_process_group."""

    logger = configure_logger(rank=config.rank)

    dist.init_process_group(**asdict(config))

    dist_info = DistributedInfo(node_rank=node_rank)
    assert dist_info.global_rank == config.rank
    assert dist_info.world_size == config.world_size

    logger.info(
        f"Initialized process group: Rank {dist_info.global_rank} out of {dist_info.world_size}."
    )

    logger.info(f"Distributed info: {dist_info}")

    display_dist_info(dist_info=dist_info, format="table", logger=logger)

    if func is not None:
        func(config.rank, config.world_size)


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


def main(world_size: int, node_rank: int) -> None:
    """Main driver function."""
    init_env(config=InitEnvArgs())
    processes = []
    mp.set_start_method("spawn")
    logger = configure_logger("main")

    for rank in range(world_size):
        init_process_group_args = InitProcessGroupArgs(rank=rank, world_size=world_size)
        p = mp.Process(
            target=init_process,
            args=(init_process_group_args, node_rank, logger, run),
            # kwargs={},
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "--world_size", default=None, type=int, help="Total number of GPUs"
    )
    parser.add_argument(
        "--node_rank", default=0, type=int, help="Node rank for multi-node training"
    )

    args = parser.parse_args()

    main(
        world_size=args.world_size if args.world_size else torch.cuda.device_count(),
        node_rank=args.node_rank,
    )
