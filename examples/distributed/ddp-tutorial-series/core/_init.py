import argparse
import logging
import os
from dataclasses import asdict
from typing import Callable, Dict, Optional

import torch.distributed as dist

from config.base import DistributedInfo, InitEnvArgs, InitProcessGroupArgs
from utils.common_utils import display_dist_info


def init_env(config: InitEnvArgs) -> None:
    """Initialize environment variables."""
    # use __dict__ to get all the attributes of the dataclass
    # if use asdict may not fetch new assigned attributes
    # after the dataclass is instantiated.
    config: Dict[str, str] = {**config.__dict__}

    for key, value in config.items():
        upper_key = key.upper()
        os.environ[upper_key] = value


def init_process(
    args: argparse.Namespace,
    init_process_group_args: InitProcessGroupArgs,
    logger: logging.Logger,
    func: Optional[Callable] = None,
) -> DistributedInfo:
    """Initialize the distributed environment via init_process_group."""

    dist.init_process_group(**asdict(init_process_group_args))

    dist_info = DistributedInfo(
        node_rank=args.node_rank,
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node,
    )

    logger.info(
        f"Initialized process group: Rank {dist_info.global_rank} "
        f"out of {dist_info.world_size}."
    )

    logger.info(f"Distributed info: {dist_info}")

    display_dist_info(dist_info=dist_info, format="table", logger=logger)

    if func is not None:
        func(init_process_group_args.rank, init_process_group_args.world_size)
    return dist_info
