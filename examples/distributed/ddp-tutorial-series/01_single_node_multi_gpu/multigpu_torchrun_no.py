"""python 01_single_node_multi_gpu/multigpu_torchrun_no.py 50 10 --node_rank 0"""
import logging
import os
from dataclasses import asdict
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from config.base import DistributedInfo, InitEnvArgs, InitProcessGroupArgs
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from utils.data_utils import ToyDataset, prepare_dataloader
from config.base import DataLoaderConfig, DistributedSamplerConfig
from utils.common_utils import configure_logger, display_dist_info, init_env, seed_all


def init_process(
    cfg: InitProcessGroupArgs,
    node_rank: int,
    logger: logging.Logger,
    func: Optional[Callable] = None,
) -> DistributedInfo:
    """Initialize the distributed environment via init_process_group."""

    dist.init_process_group(**asdict(cfg))

    dist_info = DistributedInfo(node_rank=node_rank)

    logger.info(
        f"Initialized process group: Rank {dist_info.global_rank} "
        f"out of {dist_info.world_size}."
    )

    logger.info(f"Distributed info: {dist_info}")

    display_dist_info(dist_info=dist_info, format="table", logger=logger)

    if func is not None:
        func(cfg.rank, cfg.world_size)
    return dist_info


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.logger = logger

        self.model = DDP(model, device_ids=[gpu_id])  # impt

        print(f"self.gpu_id={self.gpu_id}")

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
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        self.logger.info(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
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
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Average Loss: {avg_loss:.8f}")
        self.logger.info(
            f"[GPU{self.global_rank}] Epoch {epoch} | Average Loss: {avg_loss:.8f}"
        )

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()  # impt
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if (
                self.gpu_id == 0 and epoch % self.save_every == 0
            ):  # impt why save at rank 0?
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = ToyDataset(num_samples=2048, num_dimensions=20, target_dimensions=1)
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def main(
    rank: int,
    world_size: int,
    node_rank: int,
    save_every: int,
    total_epochs: int,
    batch_size: int,
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

    trainer = Trainer(model, train_data, optimizer, rank, save_every, logger)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    seed_all(0)

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

    args = parser.parse_args()

    # [GPU0] Epoch 0 | Batchsize: 32 | Steps: 16 because 32x16x4 gpus  = 2048 total data
    # impt below
    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(
            world_size,
            args.node_rank,
            args.save_every,
            args.total_epochs,
            args.batch_size,
        ),
        nprocs=world_size,
    )
