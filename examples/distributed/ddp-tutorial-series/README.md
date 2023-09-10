# Distributed Data Parallel

## Introduction: Distributed Data Parallel in PyTorch

- https://pytorch.org/tutorials/beginner/ddp_series_intro.html

## What is Distributed Data Parallel?

- https://pytorch.org/tutorials/beginner/ddp_series_theory.html

## Multi GPU Training with DDP

See `01_single_node_multi_gpu.py` for a simple example of multi GPU training with DDP.

- https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html

## Fault Tolerance Distributed Training with Torch Distributed Elastic

- `torchrun` and `torch.distributed.launch`.

- https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html

## Distributed without Torchrun

When you're launching a distributed training script across multiple nodes or GPUs, the coordination of when each script/process starts is crucial. The processes do not need to start simultaneously down to the millisecond, but they should all begin their work roughly at the same time and be aware of each other to effectively synchronize and communicate.

Here's a step-by-step breakdown of the process:

1. **Initialization**: Before the actual training starts, there's an initialization phase. During this phase, each process initializes its communication backend (like NCCL or Gloo for PyTorch).

2. **Master Address & Port**: One of these nodes is designated as the "master" (usually the one where the rank-0 process runs). The `master_addr` and often a `master_port` are used to establish a rendezvous, i.e., a meeting point for all processes. The other nodes will connect to this master using this address and port.

3. **Rendezvous**: When you call `torch.distributed.init_process_group()`, each process will attempt to rendezvous with the other processes. This involves some handshaking and coordination to ensure every process is accounted for. The process group will not be fully initialized until all processes have joined. Once they all join, they move forward.

4. **Barrier**: Once initialized, processes typically have a synchronization point called a barrier. If a process reaches the barrier, it waits until all processes reach this point, ensuring that all processes start the actual computation at roughly the same time.

5. **Training Loop**: After the barrier, the training loop starts. Each process computes its forward and backward passes independently on its batch of data. When it's time to update the model, gradients are synchronized across all processes (averaged). Each process then updates its local model with these averaged gradients. This ensures that all processes always have identical model parameters.

6. **Regular Synchronizations**: During training, synchronizations happen frequently, mainly during gradient averaging. Other sync points might include computation of validation metrics.

Regarding starting the scripts, if you're using a job scheduler (like SLURM, PBS, etc.), the scheduler typically handles the coordination of launching scripts on each node. If launching manually, you would start the processes nearly simultaneously on each node using scripts or SSH commands. Since they all attempt to rendezvous at the master address, they'll effectively wait for each other during initialization, ensuring coordination.

In practice, there's some flexibility. Processes don't have to start at the exact same time, but if a process is lagging too far behind or fails to start, the others will hang indefinitely during the rendezvous or barrier because they're waiting for all processes to join in.

## Question on old torch.distributed.launch

correct me if i am wrong, you mean in the argparse code below, i do not need to manually set --local-rank when running the script?
```
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int)
args = parser.parse_args()

local_rank = args.local_rank
```
---

You are correct. When you use `torch.distributed.launch`, it will **automatically** assign a value to `--local-rank` for each of the spawned processes.

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
Each of those commands runs in its separate process. In this scenario, the `--local-rank` argument is automatically appended by the launcher, and in your script, you can access this value using the argparse code you provided.

So, when you use `torch.distributed.launch`, you don't need to manually provide the `--local-rank` argument when invoking your script; the launcher will handle it for you.