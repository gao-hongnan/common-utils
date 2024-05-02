# Distributed Data Parallel

## Introduction: Distributed Data Parallel in PyTorch

-   <https://pytorch.org/tutorials/beginner/ddp_series_intro.html>

## What is Distributed Data Parallel?

-   <https://pytorch.org/tutorials/beginner/ddp_series_theory.html>

## Multi GPU Training with DDP

See `01_single_node_multi_gpu.py` for a simple example of multi GPU training
with DDP.

-   <https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html>

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

Here's a step-by-step breakdown of the process:

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
