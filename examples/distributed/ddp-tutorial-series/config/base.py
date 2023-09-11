from dataclasses import dataclass, field, fields, MISSING
from typing import Literal, Iterable, Union
import socket

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class DataLoaderConfig:
    """Configuration for data loader."""

    batch_size: int = field(
        default=32, metadata={"help": "Number of samples per batch."}
    )
    num_workers: int = field(
        default=0, metadata={"help": "Number of subprocesses to use for data loading."}
    )
    pin_memory: bool = field(
        default=True,
        metadata={
            "help": "Whether to copy tensors into CUDA pinned memory. Set it to True if using GPU."
        },
    )
    shuffle: bool = field(
        default=False,
        metadata={
            "help": "Whether to shuffle the data. Set it to False if using DistributedSampler."
        },
    )

    sampler: Union[Sampler, Iterable, None] = field(
        default=None, metadata={"help": "Sampler."}
    )


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class DistributedSamplerConfig:
    """Configuration for distributed sampler."""

    num_replicas: int
    rank: int

    shuffle: bool = field(
        default=False,
        metadata={
            "help": "Whether to shuffle the data. Set it to False if using DistributedSampler."
        },
    )
    seed: int = field(
        default=0,
        metadata={
            "help": "Random seed used to shuffle the data. Set it to 0 if using DistributedSampler."
        },
    )
    drop_last: bool = field(
        default=False,
        metadata={
            "help": "Whether to drop the last incomplete batch of data. Set it to False if using DistributedSampler."
        },
    )


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class InitEnvArgs:
    """Initialize environment variables. The attribute must be
    named such that the upper case version is the same as the
    environment variable name.
    """

    master_addr: str = field(
        default="localhost",
        metadata={
            "help": (
                "This refers to the IP address (or hostname) of the machine or node "
                "where the rank 0 process is running. It acts as the reference point "
                "for all other nodes and GPUs in the distributed setup. All other "
                "processes will connect to this address for synchronization and "
                "communication."
            )
        },
    )

    master_port: str = field(
        default="12356",
        metadata={
            "help": "Denotes an available port on the `MASTER_ADDR` machine. "
            "All processes will use this port number for communication."
        },
    )


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class InitProcessGroupArgs:
    """From torch/distributed/distributed_c10d.py:
    There are 2 main ways to initialize a process group:
    1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
    2. Specify ``init_method`` (a URL string) which indicates where/how
        to discover peers. Optionally specify ``rank`` and ``world_size``,
        or encode all required parameters in the URL and omit them.

    If neither is specified, ``init_method`` is assumed to be "env://".
    """

    rank: int = field(default=-1, metadata={"help": "Rank of the current process"})
    world_size: int = field(
        default=-1, metadata={"help": "Number of processes participating in the job"}
    )
    backend: Literal["mpi", "gloo", "nccl", "ucc"] = field(
        default="nccl", metadata={"help": "Name of the backend to use"}
    )
    init_method: str = field(
        default="env://",
        metadata={"help": "URL specifying how to initialize the package"},
    )


# frozen=True means that the class is immutable but in actual fact
# the attributes are not.
# TODO: not taking into account of group argument in dist.xxx(group=group)
# TODO: num_nodes is hard to get programmatically
@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class DistributedInfo:
    """Information about the distributed environment."""

    node_rank: int = field(default=MISSING, metadata={"help": "Rank of the node."})
    num_nodes: int = field(default=MISSING, metadata={"help": "Number of nodes."})
    num_gpus_per_node: int = field(
        default=MISSING,
        metadata={
            "help": "Number of processes per node."
            "Usually this is the number of GPUs per node."
            "Typically, each node has the same number of GPUs."
        },
    )

    num_gpus_in_curr_node_rank: int = field(
        default_factory=torch.cuda.device_count,
        metadata={"help": "Number of GPUs available on the current node."},
    )

    is_dist_available: bool = field(
        default_factory=dist.is_available,
        metadata={"help": "Whether the distributed package is initialized."},
    )
    is_dist_initialized: bool = field(
        default_factory=dist.is_initialized,
        metadata={"help": "Whether the distributed package is initialized."},
    )
    global_rank: int = field(
        default_factory=dist.get_rank,
        metadata={
            "help": "Global rank of the process in the distributed setup, "
            " can also be calculated via node_rank * num_gpus_in_curr_node_rank + local_rank "
            "where node_rank index starts from 0."
        },
    )
    world_size: int = field(
        default_factory=dist.get_world_size,
        metadata={
            "help": "Total number of processes participating in the distributed setup."
        },
    )

    local_rank: int = field(
        init=False,
        metadata={
            "help": "Local rank of the process, computed based on rank and GPUs per node."
        },
    )  # Will be computed based on rank and num_gpus_in_curr_node_rank
    device: torch.device = field(
        init=False, metadata={"help": "The torch device, either CPU or specific GPU."}
    )  # Will be computed based on local_rank

    node_hostname: str = field(
        default_factory=socket.gethostname, metadata={"help": "Hostname of the node."}
    )

    def __post_init__(self) -> None:
        # FIXME: local rank won't work if 1 process spans across multiple gpus.
        # FIXME: if for whatever reason num_gpus_in_curr_node_rank is not the
        # same as num_processes_per_node, then local_rank will be wrong.
        if self.num_processes_per_node != self.num_gpus_in_curr_node_rank:
            self.local_rank = self.global_rank % self.num_processes_per_node
        else:
            self.local_rank = self.global_rank % self.num_gpus_in_curr_node_rank

        self.device = torch.device(f"cuda:{self.local_rank}")

    @property
    def is_master(self) -> bool:
        """Check if the current rank is the master (rank 0)."""
        return self.global_rank == 0

    def get_help(self, field_name: str) -> str:
        """Retrieve the help metadata for a specific field."""
        for f in fields(self):
            if f.name == field_name:
                return f.metadata.get("help", "")
        return ""
