"""
NOTE: I think the original torch code is not entirely efficient
since it saves on the local rank of each node.

1. hostname to get current node name.
2. hostname -i to get current node ip. set this to MASTER_ADDR.
3. MASTER_PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
2. cat $PBS_NODEFILE to get all nodes.
3. ssh into the other nodes that are not the master node (step 1.)

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
import logging
import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config.base import (DataLoaderConfig, DistributedInfo,
                         DistributedSamplerConfig, InitEnvArgs,
                         InitProcessGroupArgs)
from core._init import init_env, init_process
from core._seed import seed_all
from utils.common_utils import calculate_global_rank, configure_logger
from utils.data_utils import ToyDataset, prepare_dataloader

# def ddp_setup():
#     init_process_group(backend="nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        args: argparse.Namespace,
        dist_info: DistributedInfo,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # self.local_rank = int(os.environ["LOCAL_RANK"])
        # self.global_rank = int(os.environ["RANK"])
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.model = model.to(self.local_rank)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.save_every = save_every
        self.logger = logger
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_loader))[0])
        print(
            f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}"
        )
        self.logger.info(
            f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}"
        )
        self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0  # Initialize total loss for the epoch
        for source, targets in self.train_loader:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source, targets)
            total_loss += batch_loss

        avg_loss = total_loss / len(
            self.train_loader
        )  # Calculate average loss for the epoch
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Average Loss: {avg_loss:.4f}")
        self.logger.info(
            f"[GPU{self.global_rank}] Epoch {epoch} | Average Loss: {avg_loss:.4f}"
        )

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            # save monolithic snapshot on global rank 0
            if self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs():
    train_dataset = ToyDataset(num_samples=2048, num_dimensions=20, target_dimensions=1)
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_dataset, model, optimizer


def main(
    rank: int,
    world_size: int,
    node_rank: int,
    nproc_per_node: int,
    save_every: int,
    total_epochs: int,
    batch_size: int,
    snapshot_path: str = "snapshot.pt",
):
    init_env(cfg=InitEnvArgs())

    cfg = InitProcessGroupArgs(rank=rank, world_size=world_size)

    logger = configure_logger(rank=rank)

    init_process(cfg, node_rank, logger)

    train_dataset, model, optimizer = load_train_objs()
    distributed_sampler_config = DistributedSamplerConfig(
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=0,
        drop_last=True,
    )
    distributed_sampler = DistributedSampler(
        dataset=train_dataset, **asdict(distributed_sampler_config)
    )
    dataloader_config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        sampler=distributed_sampler,
    )
    train_loader = prepare_dataloader(train_dataset, cfg=dataloader_config)

    global_rank = node_rank * nproc_per_node + rank
    logger = configure_logger(rank=rank)
    trainer = Trainer(
        model,
        train_loader,
        optimizer,
        save_every,
        snapshot_path,
        logger,
        local_rank=rank,
        global_rank=global_rank,
    )
    trainer.train(total_epochs)
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

    # Trainer
    parser.add_argument(
        "--total_epochs", default=50, type=int, help="Total epochs to train the model"
    )
    parser.add_argument(
        "--save_every", default=10, type=int, help="How often to save a snapshot"
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
    init_env_args = InitEnvArgs(
        master_addr=args.master_addr, master_port=args.master_port
    )

    mp.spawn(
        main,
        # see args=(fn, i, args, error_queue) in start_processes
        # where i is the local rank which is derived from nprocs
        args=(args, init_env_args),
        nprocs=args.num_gpus_per_node,
        join=True,
        daemon=False,
        start_method="spawn",
    )
