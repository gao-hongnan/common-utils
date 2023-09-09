from dataclasses import dataclass, field
from typing import Literal


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class InitEnvArgs:
    """Initialize environment variables. The attribute must be
    named such that the upper case version is the same as the
    environment variable name.
    """

    master_addr: str = field(
        default="localhost", metadata={"help": "IP address of the machine"}
    )
    master_port: str = field(default="12356", metadata={"help": "The port number"})


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
