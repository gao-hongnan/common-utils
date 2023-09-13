"""
export PYTHONPATH=$PYTHONPATH:$(pwd) && \
export MASTER_ADDR=$(hostname -i) && \
export MASTER_PORT=$(comm -23 <(seq 1 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1) && \
python 01_single_node_multi_gpu/multigpu_torchrun_no.py \
    --node_rank 0 \
    --num_nodes 1 \
    --num_gpus_per_node 4 \
    --world_size 4 \
    --backend "nccl" \
    --init_method "env://" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --total_epochs 50 \
    --save_every 10 \
    --batch_size 32
"""
import argparse
import logging
from dataclasses import asdict
from typing import Optional

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config.base import (
    DataLoaderConfig,
    DistributedInfo,
    DistributedSamplerConfig,
    InitEnvArgs,
    InitProcessGroupArgs,
)
from core._init import init_env, init_process
from core._seed import seed_all
from data.toy_dataset import ToyDataset, prepare_dataloader
from utils.common_utils import calculate_global_rank, configure_logger


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
        loss = F.mse_loss(output, targets)
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
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            batch_loss = self._run_batch(source, targets)
            total_loss += batch_loss

        avg_loss = total_loss / len(
            self.train_data
        )  # Calculate average loss for the epoch
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Average Loss: {avg_loss:.8f}")
        self.logger.info(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Average Loss: {avg_loss:.8f}"
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


def main(local_rank: int, args: argparse.Namespace, init_env_args: InitEnvArgs) -> None:
    # NOTE: seeding must be done inside the process, not outside for ddp.
    seed_all(0, seed_torch=True)

    global_rank = calculate_global_rank(
        local_rank=local_rank,
        node_rank=args.node_rank,
        num_gpus_per_node=args.num_gpus_per_node,
    )

    logger = configure_logger(rank=global_rank)  # unique rank across all nodes

    init_env(init_env_args)

    init_process_group_args = InitProcessGroupArgs(
        rank=global_rank,  # not local_rank
        world_size=args.world_size,
        backend=args.backend,
        init_method=args.init_method,
    )
    dist_info: DistributedInfo = init_process(
        args=args, init_process_group_args=init_process_group_args, logger=logger
    )

    dataset, model, optimizer = load_train_objs()
    distributed_sampler_config = DistributedSamplerConfig(
        num_replicas=args.world_size,
        rank=global_rank,  # this should be global rank
        shuffle=True,
        seed=0,
        drop_last=True,
    )
    distributed_sampler = DistributedSampler(
        dataset=dataset, **asdict(distributed_sampler_config)
    )
    dataloader_config = DataLoaderConfig(
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        sampler=distributed_sampler,
    )
    train_data = prepare_dataloader(dataset, cfg=dataloader_config)

    # here local rank because "we are splitting trainer across gpus"
    trainer = Trainer(model, train_data, optimizer, local_rank, args.save_every, logger)
    trainer.train(args.total_epochs)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # TODO: technically here we use hydra to parse to dataclass or pydantic, but
    # argparse is used for simplicity.
    init_env_args = InitEnvArgs(
        master_addr=args.master_addr, master_port=args.master_port
    )

    # [GPU0] Epoch 0 | Batchsize: 32 | Steps: 16 because 32x16x4 gpus  = 2048 total data
    # impt below
    # world_size = torch.cuda.device_count()
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
