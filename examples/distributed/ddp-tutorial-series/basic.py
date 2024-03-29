"""
qsub -I -l select=1:ngpus=4 -P <project_name> -l walltime=24:00:00 -q <queue_name>
module load cuda/<cuda_version> or module load cuda for latest version

cd examples/distributed/ddp-tutorial-series

## Single Node Multi GPU

export PYTHONPATH=$PYTHONPATH:$(pwd) && \
export MASTER_ADDR=$(hostname -i) && \
export MASTER_PORT=$(comm -23 <(seq 1 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1) && \
python basic.py \
    --node_rank 0 \
    --num_nodes 1 \
    --num_gpus_per_node 4 \
    --world_size 4 \
    --backend "nccl" \
    --init_method "env://" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT

and compare with:

export PYTHONPATH=$PYTHONPATH:$(pwd) && \
export MASTER_ADDR=$(hostname -i) && \
export MASTER_PORT=$(comm -23 <(seq 1 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1) && \
python basic.py \
    --node_rank 0 \
    --num_nodes 1 \
    --num_gpus_per_node 4 \
    --world_size 4 \
    --backend "nccl" \
    --init_method "env://" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --no_world_size_in_init_process

## Multi Node Multi GPU
Assume you have 2 nodes:

On Node 0:

export MASTER_ADDR=$(hostname -i) && \
export MASTER_PORT=$(comm -23 <(seq 1 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
echo $MASTER_ADDR
echo $MASTER_PORT

export PYTHONPATH=$PYTHONPATH:$(pwd) && \
export NODE_RANK=0 && \
export NUM_NODES=2 && \
export NUM_GPUS_PER_NODE=2 && \
export WORLD_SIZE=4 && \
python basic.py \
    --node_rank $NODE_RANK \
    --num_nodes $NUM_NODES \
    --num_gpus_per_node $NUM_GPUS_PER_NODE \
    --world_size $WORLD_SIZE \
    --backend "nccl" \
    --init_method "env://" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT

On Node 1:

export PYTHONPATH=$PYTHONPATH:$(pwd) && \
export MASTER_ADDR=THE_IP_ADDRESS_DEFINED_IN_NODE_0 && \
export MASTER_PORT=THE_MASTER_PORT_DEFINED_IN_NODE_0 && \
export NODE_RANK=1 && \
export NUM_NODES=2 && \
export NUM_GPUS_PER_NODE=2 && \
export WORLD_SIZE=4 && \
python basic.py \
    --node_rank $NODE_RANK \
    --num_nodes $NUM_NODES \
    --num_gpus_per_node $NUM_GPUS_PER_NODE \
    --world_size $WORLD_SIZE \
    --backend "nccl" \
    --init_method "env://" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"""
import argparse

import torch
import torch.multiprocessing as mp
from config.base import DistributedInfo, InitEnvArgs, InitProcessGroupArgs
from core._init import init_env, init_process
from utils.common_utils import calculate_global_rank, configure_logger


def main(local_rank: int, args: argparse.Namespace, init_env_args: InitEnvArgs) -> None:
    """Main driver function."""
    assert args.world_size == args.num_nodes * args.num_gpus_per_node

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
    torch.cuda.set_device(local_rank)

    assert dist_info.global_rank == global_rank
    assert dist_info.local_rank == local_rank


def main_without_world_size_in_init_process(
    local_rank: int, args: argparse.Namespace, init_env_args: InitEnvArgs
) -> None:
    """Main driver function."""
    assert args.world_size == args.num_nodes * args.num_gpus_per_node

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
        # world_size=args.world_size,
        backend=args.backend,
        init_method=args.init_method,
    )
    logger.info(f"Process group arguments: {str(init_process_group_args)}")
    dist_info: DistributedInfo = init_process(
        args=args, init_process_group_args=init_process_group_args, logger=logger
    )
    torch.cuda.set_device(local_rank)

    assert dist_info.global_rank == global_rank
    assert dist_info.local_rank == local_rank


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

    parser.add_argument("--no_world_size_in_init_process", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # TODO: technically here we use hydra to parse to dataclass or pydantic, but
    # argparse is used for simplicity.
    init_env_args = InitEnvArgs(
        master_addr=args.master_addr, master_port=args.master_port
    )
    # the rank "spawned" by mp is the local rank.
    # you can use for loop also, see torch/multiprocessing/spawn.py
    if not args.no_world_size_in_init_process:
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
    else:
        init_env_args.world_size = str(args.world_size)
        mp.spawn(
            main_without_world_size_in_init_process,
            args=(args, init_env_args),
            nprocs=args.num_gpus_per_node,
            join=True,
            daemon=False,
            start_method="spawn",
        )
