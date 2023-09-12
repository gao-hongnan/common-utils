"""
NOTE: I think the original torch code is not entirely efficient
since it saves on the local rank of each node.

1. hostname to get current node name.
2. hostname -i to get current node ip. set this to MASTER_ADDR.
3. MASTER_PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
2. cat $PBS_NODEFILE to get all nodes.
3. ssh into the other nodes that are not the master node (step 1.)

# On Node 0:

export MASTER_ADDR=$(hostname -i) && \
export MASTER_PORT=$(comm -23 <(seq 1 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
echo $MASTER_ADDR
echo $MASTER_PORT

export PYTHONPATH=$PYTHONPATH:$(pwd) && \
export NODE_RANK=0 && \
export NUM_NODES=2 && \
export NUM_GPUS_PER_NODE=2 && \
export WORLD_SIZE=4 && \
python 02_multi_node_multi_gpu/multinode_multigpu_torchrun_no.py \
    --node_rank $NODE_RANK \
    --num_nodes $NUM_NODES \
    --num_gpus_per_node $NUM_GPUS_PER_NODE \
    --world_size $WORLD_SIZE \
    --backend "nccl" \
    --init_method "env://" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --seed 0 \
    --num_workers 0 \
    --pin_memory True \
    --shuffle False \
    --drop_last False \
    --sampler_shuffle True \
    --sampler_drop_last True \
    --max_epochs 50 \
    --save_checkpoint_interval 10 \
    --batch_size 32 \
    --snapshot_path "snapshot.pt"

# On Node 1:

export PYTHONPATH=$PYTHONPATH:$(pwd) && \
export NODE_RANK=1 && \
export NUM_NODES=2 && \
export NUM_GPUS_PER_NODE=2 && \
export WORLD_SIZE=4 && \
python 02_multi_node_multi_gpu/multinode_multigpu_torchrun_no.py \
    --node_rank $NODE_RANK \
    --num_nodes $NUM_NODES \
    --num_gpus_per_node $NUM_GPUS_PER_NODE \
    --world_size $WORLD_SIZE \
    --backend "nccl" \
    --init_method "env://" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --seed 0 \
    --num_workers 0 \
    --pin_memory True \
    --shuffle False \
    --drop_last False \
    --sampler_shuffle True \
    --sampler_drop_last True \
    --max_epochs 50 \
    --save_checkpoint_interval 10 \
    --batch_size 32 \
    --snapshot_path "snapshot.pt"

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
import argparse
import functools
import logging
import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from config.base import (
    DataLoaderConfig,
    DistributedInfo,
    DistributedSamplerConfig,
    InitEnvArgs,
    InitProcessGroupArgs,
    TrainerConfig,
)
from core._init import init_env, init_process
from core._seed import seed_all
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.common_utils import calculate_global_rank, configure_logger
from utils.data_utils import ToyDataset, prepare_dataloader

# def ddp_setup():
#     init_process_group(backend="nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class Trainer:
    __slots__ = [
        "local_rank",
        "global_rank",
        "model",
        "optimizer",
        "train_loader",
        "dist_info",
        "logger",
        "epochs_run",
    ]
    local_rank: int
    global_rank: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: DataLoader
    dist_info: DistributedInfo
    logger: Optional[logging.Logger]
    epochs_run: int

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        trainer_config: TrainerConfig,
        dist_info: DistributedInfo,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.local_rank = dist_info.local_rank  # int(os.environ["LOCAL_RANK"])
        self.global_rank = dist_info.global_rank

        self.train_loader = train_loader
        self.optimizer = optimizer

        self.logger = logger

        self.epochs_run = 0

        if os.path.exists(trainer_config.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(trainer_config.snapshot_path)

        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.mse_loss(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch: int) -> None:
        batch_size = len(next(iter(self.train_loader))[0])

        self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0  # Initialize total loss for the epoch
        for source, targets in self.train_loader:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source, targets)
            total_loss += batch_loss

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(self.train_loader)

        print(
            (
                f"[GPU{self.global_rank}] Epoch {epoch} | "
                f"Batchsize: {batch_size} | Steps: {len(self.train_loader)} | "
                f"Average Loss: {avg_loss:.4f}"
            )
        )
        self.logger.info(
            (
                f"[GPU{self.global_rank}] Epoch {epoch} | "
                f"Batchsize: {batch_size} | Steps: {len(self.train_loader)} | "
                f"Average Loss: {avg_loss:.4f}"
            )
        )

    def _save_snapshot(self, epoch: int) -> None:
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.trainer_config.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.trainer_config.snapshot_path}")

    def train(self, max_epochs: int) -> None:
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            # save monolithic snapshot on global rank 0
            if self.global_rank == 0 and epoch % self.trainer_config.save_checkpoint_interval == 0:
                self._save_snapshot(epoch)


def load_train_objs():
    train_dataset = ToyDataset(num_samples=2048, num_dimensions=20, target_dimensions=1)
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_dataset, model, optimizer


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

    logger = configure_logger(rank=global_rank)  # unique rank across all nodes

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

    train_dataset, model, optimizer = load_train_objs()
    distributed_sampler_config = partial_distributed_sampler_config(rank=global_rank)
    distributed_sampler = DistributedSampler(
        dataset=train_dataset, **asdict(distributed_sampler_config)
    )

    dataloader_config = partial_dataloader_config(sampler=distributed_sampler)
    train_loader = prepare_dataloader(train_dataset, cfg=dataloader_config)

    logger = configure_logger(rank=global_rank)  # unique rank across all nodes

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        trainer_config=trainer_config,
        dist_info=dist_info,
        logger=logger,
    )
    trainer.train(max_epochs=trainer_config.max_epochs)
    destroy_process_group()


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

    # Seed
    parser.add_argument("--seed", default=0, type=int, help="Seed for reproducibility")

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

    # Trainer

    parser.add_argument(
        "--max_epochs", default=50, type=int, help="Total epochs to train the model"
    )
    parser.add_argument(
        "--save_checkpoint_interval",
        default=10,
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
        "--snapshot_path", default="snapshot.pt", type=str, help="Path to save snapshot"
    )
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
        save_checkpoint_interval=args.save_checkpoint_interval,
        batch_size=args.batch_size,
        snapshot_path=args.snapshot_path,
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
