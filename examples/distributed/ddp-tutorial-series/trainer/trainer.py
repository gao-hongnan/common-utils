from __future__ import annotations

import gc
import logging
import os
from typing import List, Optional, Tuple, Union

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.base import DistributedInfo, TrainerConfig
from utils.common_utils import configure_logger


# TODO: Consider using composer's State to maintain the state of the trainer.
# TODO: Consider using observer pattern to have on_batch_end, on_epoch_end, etc.
# pylint: disable=missing-function-docstring,missing-class-docstring
class Trainer:
    """Trainer class for training a model."""

    __slots__ = [
        "local_rank",
        "global_rank",
        "model",
        "criterion",
        "optimizer",
        "scheduler",
        "train_loader",
        "valid_loader",
        "trainer_config",
        "dist_info",
        "logger",
        "logger_all_reduce",
        "epochs_run",
        "output_dir",
        "save_path",
    ]
    local_rank: int
    global_rank: int

    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler

    train_loader: DataLoader
    valid_loader: Optional[DataLoader]

    trainer_config: TrainerConfig

    dist_info: DistributedInfo

    logger: Optional[logging.Logger]
    logger_all_reduce: logging.Logger

    epochs_run: int
    output_dir: str
    save_path: str

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,  # hard to type hint _Loss
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        trainer_config: TrainerConfig,
        dist_info: DistributedInfo,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.local_rank = dist_info.local_rank  # int(os.environ["LOCAL_RANK"])
        self.global_rank = dist_info.global_rank  # int(os.environ["RANK"])

        self.model = model.to(
            device=self.local_rank,
            dtype=next(model.parameters()).dtype,
            non_blocking=True,
        )
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.trainer_config = trainer_config

        self.dist_info = dist_info

        self.logger = logger
        self.logger_all_reduce = configure_logger(
            rank="all_reduce", print_to_console=True
        )

        self.epochs_run = 0
        self.output_dir = self._determine_output_dir()
        if self.global_rank == 0 and not os.path.exists(self.output_dir):
            # creates ./run_id/ folder
            os.makedirs(self.output_dir, exist_ok=True)

        # NOTE: To ensure only one process (usually rank 0) creates the
        # directory and others wait till it's done.
        # You must call this on all processes, not just rank 0.
        # If you put in the if clause above, the barrier is not symmetrically
        # executed by all processes because the issue is that the other processes are
        # not hitting the barrier at all.
        torch.distributed.barrier()

        # Resume from snapshot
        if trainer_config.load_path is not None and os.path.exists(
            trainer_config.load_path
        ):
            # NOTE: this is to ensure that all processes wait for each other
            # to load before proceeding to training or what not.
            torch.distributed.barrier()

            # NOTE: in DDP you would need to load the snapshot
            # on every local rank, not just rank 0.
            logger.info(f"Loading snapshot from {trainer_config.load_path}")
            map_location = f"cuda:{self.local_rank}"
            self._load_snapshot(trainer_config.load_path, map_location=map_location)

        # NOTE: only load model to DDP after loading snapshot
        # TODO: this is because in _load_snapshot we need to load the model
        # state dict, which is not DDP. Alternatively, we can load
        # snapshot["MODEL_STATE"].module in sync with the save of the snapshot.
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _determine_output_dir(self) -> str:
        """Determine the output directory based on trainer configuration."""
        return self.trainer_config.output_dir or self.trainer_config.run_id

    def _get_current_lr_or_lrs(self) -> Union[float, List[float]]:
        """Get current learning rate."""
        if len(self.optimizer.param_groups) == 1:
            return self.optimizer.param_groups[0]["lr"]

        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group["lr"])
        return lrs

    def _save_snapshot(self, epoch: int, batch: Optional[int] = None) -> None:
        """Save snapshot of the model, optimizer, and other training states."""
        if batch is not None:
            checkpoint_dir = os.path.join(
                self.output_dir, f"epoch_{epoch}_batch_{batch}"
            )
        else:
            batch = -1  # using -1 to indicate the entire epoch
            checkpoint_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.save_path = os.path.join(checkpoint_dir, "snapshot.pt")

        # TODO: consider renaming snapshot to states
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict(),
            "TORCH_RNG_STATE": torch.get_rng_state(),
            "EPOCHS_RUN": epoch,
            "BATCH_INDEX": batch,
        }
        torch.save(snapshot, self.save_path)

        self.logger.info(
            f"Epoch {epoch} Batch {batch} | Training snapshot saved at {self.save_path}"
        )

    def _load_snapshot(self, load_path: str, map_location: str) -> None:
        snapshot = torch.load(load_path, map_location=map_location)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.logger.info(f"Resuming training from snapshot at Epoch {self.epochs_run}")

        del snapshot
        gc.collect()
        torch.cuda.empty_cache()

    def _run_train_batch(
        self, source: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        return output, loss

    def _run_valid_batch(
        self, source: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():  # Disable gradient computation
            output = self.model(source)
            loss = self.criterion(output, targets)
        return output, loss

    def _run_train_epoch(self, epoch: int) -> None:
        self.model.train()  # Switch the model to train mode

        self.train_loader.sampler.set_epoch(epoch)

        total_epoch_loss = 0.0  # Initialize total loss for the epoch
        total_samples = 0

        # Initialize tqdm for rank 0 only
        train_progress_bar = (
            tqdm(
                enumerate(self.train_loader, start=1),
                total=len(self.train_loader),
                desc=f"Epoch {epoch}",
            )
            if self.global_rank == 0
            else enumerate(self.train_loader, start=1)
        )
        # batch_index starts from 1
        for _batch_index, (source, targets) in train_progress_bar:
            source = source.to(
                self.local_rank,
                non_blocking=False,
                copy=False,
                memory_format=torch.preserve_format,
            )
            targets = targets.to(
                self.local_rank,
                non_blocking=False,
                copy=False,
                memory_format=torch.preserve_format,
            )
            batch_size = source.size(0)
            _, avg_batch_loss = self._run_train_batch(source, targets)
            total_epoch_loss += avg_batch_loss * batch_size
            total_samples += batch_size

            # Update tqdm progress bar if on rank 0
            if self.global_rank == 0:
                train_progress_bar.set_description(
                    f"Epoch {epoch}, Avg Batch Loss: {avg_batch_loss.item():.4f}"
                )

        # Calculate average loss for the epoch per sample
        avg_epoch_loss = total_epoch_loss / total_samples

        # NOTE: do an all reduce to get the average loss across all processes
        # NOTE: this is not the same as the average loss across all samples
        # NOTE: so this means if I were to train on 2 nodes of 2 gpus each (4 gpus)
        # NOTE: the average loss across all processes will be the same as if I
        # NOTE: were to train on 1 node of 1 gpu (1 gpu) with the same effective batch size.
        avg_epoch_loss_all_reduce = (
            avg_epoch_loss.clone()
        )  # NOTE: expensive ops, don't do this in production
        torch.distributed.all_reduce(
            avg_epoch_loss_all_reduce, op=torch.distributed.ReduceOp.SUM
        )

        if self.dist_info.global_rank == 0:
            world_size = self.dist_info.world_size
            avg_epoch_loss_all_reduce /= world_size
            self.logger_all_reduce.info(
                f"TRAIN: Epoch {epoch} | [AVG_EPOCH_LOSS_ALL_REDUCE]: {avg_epoch_loss_all_reduce:.4f}"
            )

        current_lr = self._get_current_lr_or_lrs()

        self.logger.info(
            (
                f"[TRAIN: NODE{self.dist_info.node_rank} GPU{self.global_rank}] "
                f"Epoch {epoch} | "
                f"Batchsize: {batch_size} | Steps: {len(self.train_loader)} | "
                f"Average Loss Per Sample: {avg_epoch_loss:.4f} | Learning Rate: {current_lr}"
            )
        )

    def _run_valid_epoch(self, epoch: int) -> None:
        self.model.eval()  # Switch the model to evaluation mode

        self.valid_loader.sampler.set_epoch(epoch)

        total_epoch_loss = 0.0  # Initialize total loss for the epoch
        total_samples = 0

        valid_progress_bar = (
            tqdm(
                enumerate(self.valid_loader, start=1),
                total=len(self.valid_loader),
                desc=f"Epoch {epoch}",
            )
            if self.global_rank == 0
            else enumerate(self.valid_loader, start=1)
        )

        # Ensure no gradient is computed, saving memory and time
        with torch.no_grad():
            for _batch_index, (source, targets) in valid_progress_bar:
                source = source.to(
                    self.local_rank,
                    non_blocking=False,
                    copy=False,
                    memory_format=torch.preserve_format,
                )
                targets = targets.to(
                    self.local_rank,
                    non_blocking=False,
                    copy=False,
                    memory_format=torch.preserve_format,
                )
                batch_size = source.size(0)
                # NOTE: it is avg_batch_loss due to criterion's reduction="mean"
                _, avg_batch_loss = self._run_valid_batch(source, targets)
                total_epoch_loss += avg_batch_loss * batch_size
                total_samples += source.size(0)

                # TODO: by right saving mechanism is usually done in the callback
                # and also based on the previous metric performance.
                if self.trainer_config.save_checkpoint_interval_batch:
                    if (self.global_rank == 0) and (
                        _batch_index
                        % self.trainer_config.save_checkpoint_interval_batch
                        == 0
                    ):
                        self._save_snapshot(epoch, batch=_batch_index)

                    torch.distributed.barrier()  # as usual, barrier after saving

        avg_epoch_loss = total_epoch_loss / total_samples

        current_lr = self._get_current_lr_or_lrs()

        self.logger.info(
            (
                f"[VALID: NODE{self.dist_info.node_rank} GPU{self.global_rank}] "
                f"Epoch {epoch} | "
                f"Batchsize: {batch_size} | Steps: {len(self.valid_loader)} | "
                f"Average Loss Per Sample: {avg_epoch_loss:.4f} | Learning Rate: {current_lr}"
            )
        )

    def fit(
        self, train_loader: DataLoader, valid_loader: Optional[DataLoader] = None
    ) -> None:
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        initial_lr = self.optimizer.param_groups[0]["lr"]
        self.logger.info(f"Starting training with Learning Rate: {initial_lr}")

        for epoch in range(self.epochs_run, self.trainer_config.max_epochs):
            self._run_train_epoch(epoch)
            if self.valid_loader is not None:
                self._run_valid_epoch(epoch)

            if self.scheduler:
                self.scheduler.step()

            # TODO: here you can have a checkpoint callback to save "best" model
            # save monolithic snapshot on global rank 0
            if (
                self.global_rank == 0
                and epoch % self.trainer_config.save_checkpoint_interval_epoch == 0
            ):
                self._save_snapshot(epoch)

            # NOTE: To ensure that the main process (usually with rank 0)
            # has finished saving before other processes potentially load or
            # continue with other operations
            torch.distributed.barrier()
