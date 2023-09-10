# Single Node Multi GPU

## 01_basic and 02_basic

We pry open the basics without using `torchrun` or `torch.distributed.launch`
and on a single node with multiple GPUs.

```bash
❯ diff 01_basic.py 02_basic.py
6c6
< python 01_single_node_multi_gpu/01_basic.py \
---
> python 01_single_node_multi_gpu/02_basic.py \
69c69
<         world_size=args.world_size,
---
>         # world_size=args.world_size,
132a133,134
>     # new args
>     init_env_args.world_size = args.world_size
```

note that in `02_basic.py` I did not specify `world_size`
in `dist.init_process_group` and instead set the environment
variable `WORLD_SIZE` to `4` (the total number of GPUs on my machine).
This is achieved by setting `init_env_args.world_size = args.world_size`
as `init_env` will set the environment variables.

Note that if you do not set this environment variable, you will get the following error:

```bash
ValueError: Error initializing torch.distributed using env:// rendezvous: environment variable WORLD_SIZE expected, but not set
```

So you have to either:

- set the environment variable `WORLD_SIZE` to the total number of GPUs on your machine
- or set `world_size` in `dist.init_process_group` to the total number of GPUs on your machine

## Caveats

1. **Calculating the Global Rank**:
    - You've used arguments `--node_rank` and `--world_size`. However, the `rank` (or global rank) for each process should be calculated based on the node rank and local rank (the GPU ID on the node).
    - The calculation would typically be:
    $$ \text{global\_rank} = \text{node\_rank} \times \text{num\_gpus\_per\_node} + \text{local\_rank} $$
    Currently, this doesn't seem to be present in your code.

2. **Calculating World Size**:
    - You've provided `--world_size` as an argument, but it might be clearer if the world size was calculated automatically based on the number of nodes and GPUs per node.
    $$ \text{world\_size} = \text{num\_nodes} \times \text{num\_gpus\_per\_node} $$

3. **Master Address and Port**:
    - For a multi-node setting, you cannot use `localhost` as the `master_addr` as it won't be accessible to other nodes. You'd need to use the IP address of the node that acts as the master node.
    - The master port (`master_port`) can be left as is, but ensure the port is open and accessible from all nodes.

4. **Node Rank Determination**:
    - You've included an argument `--node_rank`. For a truly distributed setting, you'd typically determine the node rank programmatically, perhaps from environment variables set by your scheduler (like PBS or SLURM). If you're manually setting the node rank, ensure each node has a unique rank.

5. **Environment Variables**:
    - `InitEnvArgs` seems to set the `MASTER_ADDR` and `MASTER_PORT`. Ensure that the necessary environment variables (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`) are correctly set before initializing the process group.

6. **Dynamic Assert Checks**:
    - There are comments indicating the need for assertions (e.g., to check the number of GPUs). Adding these checks would make your script more robust.

To directly answer your question, while the script might function with some configurations, it lacks certain elements crucial for general multi-node, multi-GPU distributed training. Integrating the mentioned points will make the script more robust and adaptable to various settings.