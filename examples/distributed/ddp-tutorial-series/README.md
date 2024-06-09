# Distributed Data Parallel

## Point-to-Point Communication

### [Send and Recv](https://pytorch.org/tutorials/intermediate/dist_tuto.html#id1)

A transfer of data from one process to another is called a point-to-point
communication. These are achieved through the `send` and `recv` functions or
their immediate counter-parts, `isend` and `irecv`.

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

Each rank effectively runs its own instance of the `run` function due to the
`mp.Process` instantiation. Here's how it works in a step-by-step manner:

1. **For Rank 0**:

    - The process with `rank=0` starts and executes `run` function.
    - The `if rank == 0:` condition is true.
    - It increments its tensor from 0 to 1.
    - It performs `dist.send` to send this tensor to ranks 1, 2, and 3.
    - At this point, it has sent the data but hasn't confirmed that the data has
      been received by other ranks.

2. **For Rank 1**:

    - A new process is spawned with `rank=1`.
    - This process runs the `run` function.
    - The `else:` clause is executed.
    - It waits to receive the tensor from `rank=0` using `dist.recv`.
    - Once received, it prints the value, confirming the data transfer.

3. **For Rank 2 and 3**:
    - Similar to `rank=1`, new processes are spawned for `rank=2` and `rank=3`.
    - They also go into the `else:` clause and wait to receive the tensor from
      `rank=0`.

The `mp.Process` initiates these separate processes, and the `dist.send` and
`dist.recv` functions handle the point-to-point data communication between these
processes. Thus, the state (tensor) of `rank=0` is successfully transferred to
ranks 1, 2, and 3.

In the above example, both processes start with a zero tensor, then process 0
increments the tensor and sends it to process 1 so that they both end up with
1.0. Notice that processes 1,2 and 3 need to allocate memory in order to store
the data it will receive.

#### What does it mean by it needs to allocate memory?

In a scenario with four processes, each process initializes its own tensor
filled with zeroes in its respective memory space. Here's how the data flows:

-   **Process 0**: Modifies its tensor to 1 and sends this updated tensor to
    Processes 1, 2, and 3.
-   **Processes 1, 2, 3**: Each has its own pre-allocated tensor initialized to
    zero. When they execute `dist.recv`, they wait for the incoming data from
    Process 0.

Upon receiving the data, each of Processes 1, 2, and 3 overwrites its initially
zero-valued tensor with the received value of 1. Each process thus ends up with
a tensor containing the value 1, but these tensors are separate instances stored
in each process's individual memory space. The operation is in-place, meaning
the pre-allocated memory for the tensors in Processes 1, 2, and 3 is directly
updated. Process 0's tensor remains at its updated value of 1 and is not
affected by the receive operations in the other processes.

> The key is that `dist.recv` performs an in-place operation, modifying the
> tensor directly. The name `tensor` refers to a location in memory, and calling
> `dist.recv` changes the value stored in that memory location for Process 1.
> After the receive operation, the tensor's value in Process 1 becomes 1,
> replacing the initial zero. This does not affect the tensor in Process 0; they
> are separate instances in separate memory spaces.

## References and Further Readings

-   <https://pytorch.org/tutorials/intermediate/dist_tuto.html>
-   <https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py>

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

## How to load only 1 shard if you trained on multiple shards?

Say you trained on 32 nodes of 8 gpus each (256 GPUs in total) and you only want
to load 1 shard.

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

## Fault Tolerance Distributed Training with Torch Distributed Elastic

-   `torchrun` and `torch.distributed.launch`.

-   <https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html>

## basic (no_world_size_in_init_process true vs false)

We pry open the basics without using `torchrun` or `torch.distributed.launch`
and on a single node with multiple GPUs.

```bash
- world_size=args.world_size,
+ # world_size=args.world_size,
-
+ init_env_args.world_size = args.world_size
```

note that if flag is I did `no_world_size_in_init_process` to be `True`, then we
do not specify `world_size` in `dist.init_process_group` and instead set the
environment variable `WORLD_SIZE` to `4` (the total number of GPUs on my
machine). This is achieved by setting
`init_env_args.world_size = args.world_size` as `init_env` will set the
environment variables.

Note that if we do not set this environment variable, we will get the following
error:

```bash
ValueError: Error initializing torch.distributed using env:// rendezvous: environment variable WORLD_SIZE expected, but not set
```

So we have to either:

-   set the environment variable `WORLD_SIZE` to the total number of GPUs on
    your machine
-   or set `world_size` in `dist.init_process_group` to the total number of GPUs
    on your machine

Yes, placing a `dist.barrier()` can still make sense even if you're not loading
a model, depending on the context. If there's a point in your distributed
training code where all processes need to be synchronized, then `dist.barrier()`
can be used to ensure that synchronization.

Here are a few scenarios where you might want synchronization:

1. **Logging**: If you're logging metrics and want to ensure that logs from
   different processes don't overlap or get intermingled, you might use a
   barrier to synchronize the processes before the logging step.

2. **Data Preparation**: If there's a point in your code where each process
   prepares some data or does some computation that other processes depend upon,
   you would want to synchronize before continuing.

3. **Resource Access**: If multiple processes are trying to access shared
   resources (like a file or database), you might want to synchronize to avoid
   conflicts.

4. **Gradient Updates**: Normally, DistributedDataParallel (DDP) in PyTorch will
   handle gradient synchronization across processes. But if you're implementing
   custom gradient synchronization or aggregation logic, you might need explicit
   synchronization.

However, it's important to be judicious in the use of barriers:

-   Overusing barriers can impact performance since processes spend time waiting
    at the synchronization points.

-   Unnecessary barriers can negate some of the speed-up benefits you get from
    distributed training.

In summary, if there's a specific reason to ensure all processes are at the same
point in your code, then a barrier makes sense. If not, adding barriers can slow
down your training without providing any benefit.

## Single-Node Multi-GPU (Without `torchrun` or `torch.distributed.launch`)

...

## Caveats of not using `torchrun` or `torch.distributed.launch`

1. **Calculating World Size**:

    - We've provided `--world_size` as an argument, but it might be clearer if
      the world size was calculated automatically based on the number of nodes
      and GPUs per node. $$ \text{world_size} = \text{num_nodes} \times
      \text{num_gpus_per_node} $$

2. **Master Address and Port**:

    - For a multi-node setting, we cannot use `localhost` as the `master_addr`
      as it won't be accessible to other nodes. You'd need to use the IP address
      of the node that acts as the master node.
    - The master port (`master_port`) can be left as is, but ensure the port is
      open and accessible from all nodes. Can use some program like `netcat` to
      check if the port is open before running the script.

3. **Node Rank Determination**:
    - We've included an argument `--node_rank`. For a truly distributed setting,
      we'd typically determine the node rank programmatically, perhaps from
      environment variables set by your scheduler (like PBS or SLURM). If we're
      manually setting the node rank, ensure each node has a unique rank.

## Distributed without Torchrun

When you're launching a distributed training script across multiple nodes or
GPUs, the coordination of when each script/process starts is crucial. The
processes do not need to start simultaneously down to the millisecond, but they
should all begin their work roughly at the same time and be aware of each other
to effectively synchronize and communicate.

1. **Initialization**: Before the actual training starts, there's an
   initialization phase. During this phase, each process initializes its
   communication backend (like NCCL or Gloo for PyTorch).

2. **Master Address & Port**: One of these nodes is designated as the "master"
   (usually the one where the rank-0 process runs). The `master_addr` and often
   a `master_port` are used to establish a rendezvous, i.e., a meeting point for
   all processes. The other nodes will connect to this master using this address
   and port.

3. **Rendezvous**: When you call `torch.distributed.init_process_group()`, each
   process will attempt to rendezvous with the other processes. This involves
   some handshaking and coordination to ensure every process is accounted for.
   The process group will not be fully initialized until all processes have
   joined. Once they all join, they move forward.

4. **Barrier**: Once initialized, processes typically have a synchronization
   point called a barrier. If a process reaches the barrier, it waits until all
   processes reach this point, ensuring that all processes start the actual
   computation at roughly the same time.

5. **Training Loop**: After the barrier, the training loop starts. Each process
   computes its forward and backward passes independently on its batch of data.
   When it's time to update the model, gradients are synchronized across all
   processes (averaged). Each process then updates its local model with these
   averaged gradients. This ensures that all processes always have identical
   model parameters.

6. **Regular Synchronizations**: During training, synchronizations happen
   frequently, mainly during gradient averaging. Other sync points might include
   computation of validation metrics.

Regarding starting the scripts, if you're using a job scheduler (like SLURM,
PBS, etc.), the scheduler typically handles the coordination of launching
scripts on each node. If launching manually, you would start the processes
nearly simultaneously on each node using scripts or SSH commands. Since they all
attempt to rendezvous at the master address, they'll effectively wait for each
other during initialization, ensuring coordination.

In practice, there's some flexibility. Processes don't have to start at the
exact same time, but if a process is lagging too far behind or fails to start,
the others will hang indefinitely during the rendezvous or barrier because
they're waiting for all processes to join in.

## Question on old torch.distributed.launch

correct me if i am wrong, you mean in the argparse code below, i do not need to
manually set --local-rank when running the script?

```
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int)
args = parser.parse_args()

local_rank = args.local_rank
```

---

You are correct. When you use `torch.distributed.launch`, it will
**automatically** assign a value to `--local-rank` for each of the spawned
processes.

For instance, if you run a script using:

```bash
python -m torch.distributed.launch --nproc_per_node=4 your_script.py --other-arg=value
```

The launcher will internally run your script as if you had invoked:

```bash
your_script.py --other-arg=value --local-rank=0
your_script.py --other-arg=value --local-rank=1
your_script.py --other-arg=value --local-rank=2
your_script.py --other-arg=value --local-rank=3
```

Each of those commands runs in its separate process. In this scenario, the
`--local-rank` argument is automatically appended by the launcher, and in your
script, you can access this value using the argparse code you provided.

So, when you use `torch.distributed.launch`, you don't need to manually provide
the `--local-rank` argument when invoking your script; the launcher will handle
it for you.

# Multi Node Multi GPU

## Seeding during Resumption

See `reproducibility.py` in composer, might need to refer to
<https://discuss.pytorch.org/t/reproducibility-with-checkpoints/104344/7>.

## Loading Checkpoints in DDP

**Background Context**: Distributed Data Parallel (DDP) in PyTorch works by
creating multiple processes, with each process handling a subset of the data on
one device (usually a GPU). Each process has its own model replica. For
consistent results, it's crucial that all model replicas start with the same
weights, whether they are initialized randomly or loaded from a checkpoint.

**Step-by-step Thinking**:

1. **Model Initialization in DDP**:

    When using DDP, the typical process is to initialize the model in each
    process and then either:

    - Broadcast the weights from one process (usually the one with rank 0) to
      all other processes, or
    - Load a checkpoint in each process independently.

2. **Loading Checkpoints**:

    If you're loading a checkpoint:

    - You need to ensure that each process loads the checkpoint independently.
      This is because each process has its own model replica and operates
      somewhat autonomously in DDP.
    - You might think that loading a checkpoint on one GPU and then broadcasting
      the weights to others might work (and technically, it would). However,
      loading the checkpoint independently on each GPU can be more efficient and
      simpler.

        ```

        if trainer_config.load_path is not None and os.path.exists(
            trainer_config.load_path
        ):
            # NOTE: in DDP you would need to load the snapshot
            # on every local rank, not just rank 0.
            logger.info(f"Loading snapshot from {trainer_config.load_path}")
            map_location = f"cuda:{self.local_rank}"
            self._load_snapshot(trainer_config.load_path, map_location=map_location)
        ```

In DDP, each process should load the snapshot for its own model replica. The
reason is, as mentioned, each process in DDP handles a separate model replica,
and for consistent results across processes, all replicas need to start with the
same weights. By loading the snapshot on every local rank, you ensure each model
replica starts with identical weights.

## Rendezvous vs Master?

The `master_addr` and `master_port` are indeed used in distributed training for
specifying where the initial communication (rendezvous) happens. This is
especially true for the `static` rendezvous backend in PyTorch's distributed
training.

To give intuition:

1. **Static Rendezvous Backend**:

    The `static` backend, as the name suggests, means the addresses and ports
    are predetermined. It's the more traditional way to set up distributed
    training, where one explicitly specifies the master node's address and port.
    This is where processes will rendezvous to set up their communication.

    - If you use the `static` backend, you need to specify the `master_addr` and
      `master_port` unless a `rdzv_endpoint` is provided.
    - For instance, if you're initiating a distributed process group with
      `torch.distributed`, you might see something like:

        ```python
        torch.distributed.init_process_group(
           backend='nccl',
           init_method='tcp://192.168.1.1:12345',  # where 192.168.1.1 is master_addr and 12345 is master_port
           rank=rank,
           world_size=world_size
        )
        ```

2. **Dynamic Rendezvous Backend**:

    Some modern approaches to distributed training utilize dynamic rendezvous
    methods. Instead of hardcoding `master_addr` and `master_port`, the system
    might discover available nodes dynamically and negotiate who does what. This
    is handy for cloud environments or kubernetes setups where you might not
    always have a fixed IP and port.

    - If you're using a dynamic rendezvous backend like `etcd`, `c10d`, etc.,
      the setup might look different. In such cases, `rdzv_endpoint` might be
      provided to point to the service handling the rendezvous.

For clarity, when using the `static` backend:

-   If `rdzv_endpoint` is **not** specified, `master_addr` and `master_port` are
    used.
-   If `rdzv_endpoint` is specified, it takes precedence over `master_addr` and
    `master_port`.

When you're setting up distributed training, ensure you're consistent in the
choice of your rendezvous method and the corresponding parameters to avoid
confusion or unexpected behaviors.
