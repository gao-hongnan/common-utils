import logging
import os
from dataclasses import asdict
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from config.base import (
    DataLoaderConfig,
    DistributedInfo,
    DistributedSamplerConfig,
    InitEnvArgs,
    InitProcessGroupArgs,
)
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from utils.common_utils import configure_logger, display_dist_info
from utils.data_utils import ToyDataset, prepare_dataloader

# def ddp_setup():
#     init_process_group(backend="nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def init_env(cfg: InitEnvArgs) -> None:
    """Initialize environment variables."""
    cfg: Dict[str, str] = asdict(cfg)

    for key, value in cfg.items():
        upper_key = key.upper()
        os.environ[upper_key] = value


def init_process(
    cfg: InitProcessGroupArgs,
    node_rank: int,
    logger: logging.Logger,
    func: Optional[Callable] = None,
) -> None:
    """Initialize the distributed environment via init_process_group."""

    logger = configure_logger(rank=cfg.rank)

    dist.init_process_group(**asdict(cfg))

    dist_info = DistributedInfo(node_rank=node_rank)
    assert dist_info.global_rank == cfg.rank
    assert dist_info.world_size == cfg.world_size

    logger.info(
        f"Initialized process group: Rank {dist_info.global_rank} out of {dist_info.world_size}."
    )

    logger.info(f"Distributed info: {dist_info}")

    display_dist_info(dist_info=dist_info, format="table", logger=logger)

    if func is not None:
        func(cfg.rank, cfg.world_size)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        logger: Optional[logging.Logger] = None,
        local_rank: int = -1,
        global_rank: int = -1,
    ) -> None:
        # self.local_rank = int(os.environ["LOCAL_RANK"])
        # self.global_rank = int(os.environ["RANK"])
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.model = model.to(self.local_rank)
        self.train_data = train_data
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
        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        self.logger.info(
            f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        self.train_data.sampler.set_epoch(epoch)

        total_loss = 0.0  # Initialize total loss for the epoch
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source, targets)
            total_loss += batch_loss

        avg_loss = total_loss / len(
            self.train_data
        )  # Calculate average loss for the epoch
        print(
            f"[GPU{self.global_rank}] Epoch {epoch} | Average Loss: {avg_loss:.4f}"
        )  # Print the average loss
        self.logger.info(
            f"[GPU{self.global_rank}] Epoch {epoch} | Average Loss: {avg_loss:.4f}"
        )  # Print the average loss

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs():
    train_set = ToyDataset(num_samples=2048, num_dimensions=20, target_dimensions=1)
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


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

    dataset, model, optimizer = load_train_objs()
    distributed_sampler_config = DistributedSamplerConfig(
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=0,
        drop_last=True,
    )
    distributed_sampler = DistributedSampler(
        dataset=dataset, **asdict(distributed_sampler_config)
    )
    dataloader_config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        sampler=distributed_sampler,
    )
    train_data = prepare_dataloader(dataset, cfg=dataloader_config)

    global_rank = node_rank * nproc_per_node + rank
    logger = configure_logger(rank=rank)
    trainer = Trainer(
        model,
        train_data,
        optimizer,
        save_every,
        snapshot_path,
        logger,
        local_rank=rank,
        global_rank=global_rank,
    )
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    parser.add_argument(
        "--node_rank", default=0, type=int, help="Node rank for multi-node training"
    )
    parser.add_argument("--nproc_per_node", default=2, type=int, help="Number of GPUs per node")

    args = parser.parse_args()

    # main(args.save_every, args.total_epochs, args.batch_size)#
    # this is local world size for 1 node
    # NOTE: mp spawns' rank is local rank.
    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(
            world_size,
            args.node_rank,
            args.nproc_per_node,
            args.save_every,
            args.total_epochs,
            args.batch_size,
        ),
        nprocs=world_size,
    )
