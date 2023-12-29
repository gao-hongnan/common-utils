"""
NOTE: I think the original torch code is not entirely efficient
since it saves on the local rank of each node.

NOTE: With 4 gpus in DDP you will have 4 Trainers!

NOTE: Effective batch size usually implies the same result ensues
if you train on all 4 gpus vs 1 gpus just by maintaining the same effective
batch size.

NOTE: The `avg_epoch_loss_all_reduce` is still an average loss per sample for the entire epoch. However, its value represents a collective average across all the participating processes. If all processes are working on roughly equal portions of the data and the data is identically distributed across all processes, then `avg_epoch_loss_all_reduce` should be very close to the `avg_epoch_loss` computed on any single process.

It becomes meaningful in a distributed setting to ensure that the training is proceeding similarly across all processes. In certain distributed settings, there can be some drift or disparity in the updates across nodes or GPUs due to various reasons (e.g., staleness in updates, different data partitions, etc.). By monitoring the `avg_epoch_loss_all_reduce`, you can ensure that all processes are on track and there's no significant divergence in the learned models across processes.

1. hostname to get current node name.
2. hostname -i to get current node ip. set this to MASTER_ADDR.
3. MASTER_PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
2. cat $PBS_NODEFILE to get all nodes.
3. ssh into the other nodes that are not the master node (step 1.)

NOTE: To resume, just load ckpt on each node with --load_path.

abstract
torchrun \
--nproc_per_node=$NUM_GPUS_PER_NODE \
--nnodes=$NUM_NODES \
--node_rank=$NODE_RANK \
--rdzv_id=$JOB_ID \
--rdzv_backend=c10d \
--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
--max_restarts=3 \

torchrun \
--nproc_per_node=2 \
--nnodes=2 \
--node_rank=0 \
--rdzv_id=123 \
--rdzv_backend=c10d \
--rdzv_endpoint=10.168.0.30:12356 \
--max_restarts=3 \
multinode.py 50 10

torchrun \
--nproc_per_node=2 \
--nnodes=2 \
--node_rank=1 \
--rdzv_id=123 \
--rdzv_backend=c10d \
--rdzv_endpoint=10.168.0.30:12356 \
--max_restarts=3 \
multinode.py 50 10
"""
from __future__ import annotations

import argparse
import copy
import functools
from dataclasses import asdict
from typing import Tuple

import torch
import torch.multiprocessing as mp
from config._criterion import build_criterion
from config._optim import build_optimizer
from config._scheduler import build_scheduler
from config.base import (
    CRITERION_NAME_TO_CONFIG_MAPPING,
    OPTIMIZER_NAME_TO_CONFIG_MAPPING,
    SCHEDULER_NAME_TO_CONFIG_MAPPING,
    DataLoaderConfig,
    DistributedInfo,
    DistributedSamplerConfig,
    InitEnvArgs,
    InitProcessGroupArgs,
    TrainerConfig,
)
from core._init import init_env, init_process
from core._seed import seed_all
from data.toy_dataset import ToyDataset, prepare_dataloader
from models.toy_model import ExtendedToyModel, ToyModel
from rich.pretty import pprint
from torch.distributed import destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from trainer.trainer import Trainer
from utils.common_utils import calculate_global_rank, configure_logger, deprecated


@deprecated("Use build_all_elegant instead.")
def build_all(
    args: argparse.Namespace,
) -> Tuple[
    ToyDataset,
    ToyModel,
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler._LRScheduler,
]:
    train_dataset = ToyDataset(
        num_samples=args.num_samples,
        num_dimensions=args.num_dimensions,
        target_dimensions=args.target_dimensions,
    )
    model = ToyModel(input_dim=args.input_dim, output_dim=args.output_dim)
    criterion = torch.nn.MSELoss(reduction=args.reduction)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = None
    return train_dataset, model, criterion, optimizer, scheduler


def build_all_elegant(
    args: argparse.Namespace,
) -> Tuple[
    ToyDataset,
    ToyModel,
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler._LRScheduler,
]:
    """A more elegant way of building the model, criterion, optimizer, and dataset
    by using design pattern"""
    train_dataset = ToyDataset(
        num_samples=args.num_samples,
        num_dimensions=args.num_dimensions,
        target_dimensions=args.target_dimensions,
    )

    if args.model_name == "toy_model":
        model = ToyModel(input_dim=args.input_dim, output_dim=args.output_dim)
    elif args.model_name == "extended_toy_model":
        model = ExtendedToyModel(
            input_dim=args.input_dim,
            hidden_dims=args.hidden_dims,
            output_dim=args.output_dim,
        )
    else:
        raise ValueError(f"Model name {args.model_name} not supported.")

    criterion_config = CRITERION_NAME_TO_CONFIG_MAPPING[args.criterion_name](
        name=args.criterion_name, reduction=args.reduction
    )
    criterion = build_criterion(config=criterion_config)

    optimizer_config = OPTIMIZER_NAME_TO_CONFIG_MAPPING[args.optimizer_name](
        name=args.optimizer_name, lr=args.lr
    )
    optimizer = build_optimizer(model=model, config=optimizer_config)

    if args.scheduler_name:
        scheduler_config = SCHEDULER_NAME_TO_CONFIG_MAPPING[args.scheduler_name](
            name=args.scheduler_name, optimizer=optimizer
        )
        scheduler = build_scheduler(config=scheduler_config)
    else:
        scheduler = None
    return train_dataset, model, criterion, optimizer, scheduler


def main(
    local_rank: int,
    args: argparse.Namespace,
    init_env_args: InitEnvArgs,
    partial_dataloader_config: DataLoaderConfig,
    partial_distributed_sampler_config: DistributedSamplerConfig,
    trainer_config: TrainerConfig,
) -> None:
    assert args.world_size == args.num_nodes * args.num_gpus_per_node

    seed_all(seed=args.seed, seed_torch=True)

    global_rank = calculate_global_rank(
        local_rank=local_rank,
        node_rank=args.node_rank,
        num_gpus_per_node=args.num_gpus_per_node,
    )

    # unique rank across all nodes for logging
    logger = configure_logger(rank=global_rank, print_to_console=args.print_to_console)
    logger.info(
        f"Initializing the following Environment variables: {str(init_env_args)}"
    )
    init_env(init_env_args)

    init_process_group_args = InitProcessGroupArgs(
        rank=global_rank,  # not local_rank
        world_size=args.world_size,
        backend=args.backend,
        init_method=args.init_method,
    )
    logger.info(f"Process group arguments: {str(init_process_group_args)}")
    dist_info: DistributedInfo = init_process(
        args=args, init_process_group_args=init_process_group_args, logger=logger
    )
    logger.info(f"Distributed info: {str(dist_info)}")
    torch.cuda.set_device(local_rank)

    # train_dataset, model, criterion, optimizer = build_all(args=args)
    train_dataset, model, criterion, optimizer, scheduler = build_all_elegant(args=args)

    distributed_sampler_config = partial_distributed_sampler_config(rank=global_rank)
    distributed_sampler = DistributedSampler(
        dataset=train_dataset, **asdict(distributed_sampler_config)
    )

    dataloader_config = partial_dataloader_config(sampler=distributed_sampler)
    train_loader = prepare_dataloader(train_dataset, config=dataloader_config)
    # NOTE: set valid loader as clone of train loader for simplicity
    # note it will produce different loss results due to model.eval() mode.
    valid_loader = copy.deepcopy(train_loader)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_config=trainer_config,
        dist_info=dist_info,
        logger=logger,
    )
    trainer.fit(train_loader=train_loader, valid_loader=valid_loader)
    if global_rank == 0:
        epoch_state = trainer.epoch_state
        history = trainer.history.states

        pprint(epoch_state)
        pprint(history)
    destroy_process_group()


# TODO: all of these configs can be instantiated from hydra and wrapped
# around dataclass/pydantic. Do compose pattern to compose the configs.
def parse_args() -> argparse.Namespace:
    """
    Parses the input arguments for the distributed training job.

    Returns
    -------
    argparse.Namespace: Parsed arguments.
    """
    # TODO: use hydra to parse arguments
    parser = argparse.ArgumentParser(description="simple distributed training job")

    parser.add_argument(
        "--node_rank", default=-1, type=int, help="Node rank for multi-node training"
    )
    parser.add_argument("--num_nodes", default=1, type=int, help="Number of nodes")
    parser.add_argument(
        "--num_gpus_per_node", default=4, type=int, help="Number of GPUs"
    )
    # FIXME: do some assert check to ensure num gpus per node is indeed the same programatically

    parser.add_argument(
        "--world_size", default=-1, type=int, help="Total number of GPUs"
    )

    # InitProcessGroupArgs
    # NOTE: rank and world_size can be inferred from node_rank and world_size,
    # so there is no need to specify them. Furthermore, it is difficult to
    # pre-define rank.
    parser.add_argument("--backend", default="nccl", type=str, help="Backend to use")
    parser.add_argument("--init_method", default="env://", type=str, help="Init method")

    # InitEnvArgs
    parser.add_argument(
        "--master_addr",
        default="localhost",
        type=str,
        help="Master address for distributed job. See InitEnvArgs for more details.",
    )
    parser.add_argument(
        "--master_port",
        default="12356",
        type=str,
        help="Master port for distributed job. See InitEnvArgs for more details.",
    )

    # Logger
    parser.add_argument(
        "--print_to_console", action="store_true", help="Print to console"
    )

    # Seed
    parser.add_argument("--seed", default=0, type=int, help="Seed for reproducibility")

    # Dataset
    parser.add_argument(
        "--num_samples", default=2048, type=int, help="Number of samples"
    )
    parser.add_argument(
        "--num_dimensions", default=20, type=int, help="Number of dimensions"
    )
    parser.add_argument(
        "--target_dimensions", default=1, type=int, help="Target dimensions"
    )

    # DataLoader
    parser.add_argument("--num_workers", default=0, type=int, help="Number of workers")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle")
    parser.add_argument(
        "--drop_last", action="store_true", help="Drop last batch if incomplete"
    )

    # DistributedSampler
    parser.add_argument("--sampler_shuffle", action="store_true", help="Shuffle")
    parser.add_argument("--sampler_drop_last", action="store_true", help="Drop last")

    # Model
    parser.add_argument("--model_name", default="toy_model", type=str, help="Model")
    parser.add_argument("--input_dim", default=20, type=int, help="Input dimensions")
    parser.add_argument("--hidden_dims", default=[128, 64], type=list, help="Hidden")
    parser.add_argument("--output_dim", default=1, type=int, help="Output dimensions")

    # Criterion
    parser.add_argument(
        "--criterion_name", default="mse_loss", type=str, help="Criterion"
    )
    parser.add_argument("--reduction", default="mean", type=str, help="Reduction")

    # Optimizer
    parser.add_argument("--optimizer_name", default="sgd", type=str, help="Optimizer")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")

    # Scheduler
    parser.add_argument("--scheduler_name", default=None, type=str, help="Scheduler")
    # add your necessary scheduler arguments here if needed, else refer to
    # config/_scheduler.py for default arguments

    # Trainer
    parser.add_argument(
        "--max_epochs", default=50, type=int, help="Total epochs to train the model"
    )
    parser.add_argument(
        "--save_checkpoint_interval_epoch",
        default=10,
        type=int,
        help="How often to save a snapshot",
    )
    parser.add_argument(
        "--save_checkpoint_interval_batch",
        default=None,
        type=int,
        help="How often to save a snapshot",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    parser.add_argument(
        "--output_dir", default=None, type=str, help="Path to save snapshot"
    )
    parser.add_argument("--load_path", default=None, type=str, help="Path to load")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # TODO: technically here we use hydra to parse to dataclass or pydantic, but
    # argparse is used for simplicity.
    init_env_args: InitEnvArgs = InitEnvArgs(
        master_addr=args.master_addr, master_port=args.master_port
    )
    partial_dataloader_config: DataLoaderConfig = functools.partial(
        DataLoaderConfig,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=args.shuffle,
        drop_last=args.drop_last,
    )
    partial_distributed_sampler_config: DistributedSamplerConfig = functools.partial(
        DistributedSamplerConfig,
        num_replicas=args.world_size,
        shuffle=args.sampler_shuffle,  # sampler shuffle is exclusive with DataLoader
        seed=args.seed,
        drop_last=args.sampler_drop_last,
    )

    trainer_config: TrainerConfig = TrainerConfig(
        max_epochs=args.max_epochs,
        save_checkpoint_interval_epoch=args.save_checkpoint_interval_epoch,
        save_checkpoint_interval_batch=args.save_checkpoint_interval_batch,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        load_path=args.load_path,
    )

    mp.spawn(
        main,
        # see args=(fn, i, args, error_queue) in start_processes
        # where i is the local rank which is derived from nprocs
        args=(
            args,
            init_env_args,
            partial_dataloader_config,
            partial_distributed_sampler_config,
            trainer_config,
        ),
        nprocs=args.num_gpus_per_node,
        join=True,
        daemon=False,
        start_method="spawn",
    )
