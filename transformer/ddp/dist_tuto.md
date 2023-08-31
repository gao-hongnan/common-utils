# WRITING DISTRIBUTED APPLICATIONS WITH PYTORCH

```python
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
from multigpu import prepare_dataloader,load_train_objs, Trainer

def configure_logger(rank: int) -> logging.Logger:
    handlers = [logging.FileHandler(filename=f"process_{rank}.log")] # , RichHandler()]
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
    """To be implemented."""
    ...


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
    parser = argparse.ArgumentParser(description='simple distributed training job')
    #parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    #parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    #parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--world_size', default=None, type=int, help='Total number of GPUs')

    args = parser.parse_args()

    if not args.world_size:
        world_size = torch.cuda.device_count()
    else:
        world_size = args.world_size

    main(world_size)
```

```python
def run(rank: int, world_size: int) -> None:
    """Blocking point-to-point communication."""
    tensor = torch.zeros(1)
    print('Rank ', rank, ' has data ', tensor[0])
    tensor = tensor.to(rank) # in 1 node, global rank = local rank
    if rank == 0:
        tensor += 1
        # Send the tensor to processes other than 0
        for other_rank in range(1, world_size):  # Sending to all other ranks
            dist.send(tensor=tensor, dst=other_rank)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)

    print('Rank ', rank, ' has data ', tensor[0])
```

Each rank effectively runs its own instance of the `run` function due to the `mp.Process` instantiation. Here's how it works in a step-by-step manner:

1. **For Rank 0**:
    - The process with `rank=0` starts and executes `run` function.
    - The `if rank == 0:` condition is true.
    - It increments its tensor from 0 to 1.
    - It performs `dist.send` to send this tensor to ranks 1, 2, and 3.
    - At this point, it has sent the data but hasn't confirmed that the data has been received by other ranks.

2. **For Rank 1**:
    - A new process is spawned with `rank=1`.
    - This process runs the `run` function.
    - The `else:` clause is executed.
    - It waits to receive the tensor from `rank=0` using `dist.recv`.
    - Once received, it prints the value, confirming the data transfer.

3. **For Rank 2 and 3**:
    - Similar to `rank=1`, new processes are spawned for `rank=2` and `rank=3`.
    - They also go into the `else:` clause and wait to receive the tensor from `rank=0`.

The `mp.Process` initiates these separate processes, and the `dist.send` and `dist.recv` functions handle the point-to-point data communication between these processes. Thus, the state (tensor) of `rank=0` is successfully transferred to ranks 1, 2, and 3.

## References and Further Readings

- https://pytorch.org/tutorials/intermediate/dist_tuto.html
- https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py