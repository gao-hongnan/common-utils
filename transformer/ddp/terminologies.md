# Terminologies

Let's consider a machine learning training job on a cluster with 2 nodes (Node A and Node B), each having 4 GPUs.

## Thread vs Process

...

## Processes
1. On Node A, you launch two processes: Process A1 controls GPU0 on Node A, and Process A2 controls GPU1 on Node A.
2. On Node B, likewise, you launch two processes: Process B1 controls GPU0 on Node B, and Process B2 controls GPU1 on Node B.

So, in total, you have 4 processes: A1, A2, B1, and B2.

Example: Let's say you're running the training of a neural network model. Process A1 will take a subset of the data, perform forward and backward passes on GPU0 of Node A, and update the model parameters. The same goes for Processes A2, B1, and B2 but on their respective GPUs.

### Process Groups
A process group is a way to logically group these processes for collective communication. Let's say you create a process group that includes all 4 processes: A1, A2, B1, and B2.

Now, when you perform a collective operation like a broadcast or a reduce, all processes in this group will participate. For example, if Process A1 computes some gradients, a reduce operation would sum these gradients across all 4 processes, and each process will receive the sum.

1. **Global (or World) Group**: This is a default group created which includes all processes. If you initiate 4 processes, the global group would consist of all these 4 processes.

2. **Custom Groups**: You can also create custom groups with a subset of all available processes. For example, you can create a group only consisting of processes on Node A (A1 and A2). This is useful for operations that only concern a subset of all processes.

Hope this example clarifies the concepts of processes and process groups in the context of distributed training.


- **World**: In distributed computing, the "world" refers to the collection of
    processes that can communicate with each other. In PyTorch, this is
    formalized as a process group, and the size of the world would be the number
    of processes in that group.

- **Rank**: Each process in the world is assigned a unique integer ID, which
    is its "rank." Ranks are contiguous and start from 0.

- **Local Rank**: When using multiple GPUs on the same node, the local rank
    specifies the GPU to use on that node. Local ranks are assigned per node,
    unlike global ranks which are unique across all nodes.

- **Global Rank**: This is the unique identifier assigned to a process in a
    multi-node distributed setting. It is unique across all GPUs across all
    nodes and is what we commonly refer to as the "rank" in distributed
    computing.

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

### Examples

#### Multi-GPU Single Node

If you have 2 GPUs on a single node, `local_rank` could be 0 or 1, specifying
which GPU you're referring to. The `global_rank` and `rank` are the same in a
single-node setting.

```python
# On GPU 0
local_rank = 0
global_rank = 0

# On GPU 1
local_rank = 1
global_rank = 1
```

#### Multi-GPU Multi-Node

If you have 2 nodes each with 2 GPUs, you could have the following setup:

```python
# Node 0, GPU 0
local_rank = 0
global_rank = 0

# Node 0, GPU 1
local_rank = 1
global_rank = 1

# Node 1, GPU 0
local_rank = 0
global_rank = 2

# Node 1, GPU 1
local_rank = 1
global_rank = 3
```

Here, `global_rank` is unique across all GPUs on all nodes, while `local_rank`
resets on each node.

These ranks and world size often determine how data and computation are
distributed across processes, and understanding them is essential for efficient
distributed training.
