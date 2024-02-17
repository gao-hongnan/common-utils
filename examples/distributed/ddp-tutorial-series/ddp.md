# Intro

## What is DDP?

<https://pytorch.org/tutorials/beginner/ddp_series_theory.html>

## Single GPU Training

Certainly, understanding the concept of ranks and the world in distributed
computing can be crucial for implementing efficient parallel algorithms. In
PyTorch, these concepts are made concrete through several utility functions in
the `torch.distributed` library. Let's dig deep into these terms:

## Multi-GPU Training with DistributedDataParallel (DDP)

## Multi-Node Multi-GPU Training with DistributedDataParallel (DDP)

```bash
torchrun --nproc_per_node=2 \
         --nnodes=2 \
         --node_rank=0 \
         --rdzv_id=my_job \
         --rdzv_backend=c10d \
         --rdzv_endpoint=10.168.0.11:29603 \
         multinode.py 50 10 --batch_size 32

torchrun --nproc_per_node=2 \
         --nnodes=2 \
         --node_rank=1 \
         --rdzv_id=my_job \
         --rdzv_backend=c10d \
         --rdzv_endpoint=10.168.0.11:29603 \
         multinode.py 50 10 --batch_size 32
```

## Model Parallelism (Can be independent from DDP)

- <https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html>

## How to load only 1 shard if you trained on multiple shards?

Say you trained on 32 nodes of 8 gpus each (256 GPUs in total) and you only
want to load 1 shard.

```python
import os
import torch

import torch.distributed as dist
from torch.distributed import _shard

os.environ['MASTER_ADDR'] = 'localhost' # or find what address you used
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group(backend='gloo', world_size=1, rank=0)


def custom_setstate(self, state):
    # Bypass the process group check
    self._sharded_tensor_id = None

    # Continue with the original logic
    self._local_shards, self._metadata, pg_state, self._sharding_spec, self._init_rrefs = state

# Replace the original __setstate__ method with the custom one
_shard.sharded_tensor.api.ShardedTensor.__setstate__ = custom_setstate

path = "/multi-node/ep0-ba20500-rank0.pt"
print(path)
shard = torch.load(path, map_location='cpu')
# print(shard['state']['model'])
print(shard['state']['schedulers'])
print(shard['state']['optimizers']["DecoupledAdamW"].keys())
print(shard['state']['optimizers']["DecoupledAdamW"]["param_groups"])
```

### Utility Methods and Examples

#### `dist.get_rank(group=None)`

Returns the rank within the group (or the world if no group is specified).

```python
# On process with rank 0
print(dist.get_rank())  # Output will be 0
```

#### `dist.get_world_size(group=None)`

Returns the number of processes in the group (or world).

```python
# In a world with 4 processes
print(dist.get_world_size())  # Output will be 4
```

#### `dist.get_backend(group=None)`

Get the backend of a process group.

```python
# After initializing with 'nccl' backend
print(dist.get_backend())  # Output will be 'nccl'
```
