# Single Node Multi GPU

## basic (no_world_size_in_init_process true vs false)

We pry open the basics without using `torchrun` or `torch.distributed.launch`
and on a single node with multiple GPUs.

```bash
- world_size=args.world_size,
+ # world_size=args.world_size,
-
+ init_env_args.world_size = args.world_size
```

note that if flag is I did `no_world_size_in_init_process` to be `True`, then
we do not specify `world_size`
in `dist.init_process_group` and instead set the environment
variable `WORLD_SIZE` to `4` (the total number of GPUs on my machine).
This is achieved by setting `init_env_args.world_size = args.world_size`
as `init_env` will set the environment variables.

Note that if we do not set this environment variable, we will get the following error:

```bash
ValueError: Error initializing torch.distributed using env:// rendezvous: environment variable WORLD_SIZE expected, but not set
```

So we have to either:

- set the environment variable `WORLD_SIZE` to the total number of GPUs on your machine
- or set `world_size` in `dist.init_process_group` to the total number of GPUs on your machine

## Single-Node Multi-GPU (Without `torchrun` or `torch.distributed.launch`)

...

## Caveats of not using `torchrun` or `torch.distributed.launch`

1. **Calculating World Size**:
    - We've provided `--world_size` as an argument, but it might be clearer if the world size was calculated automatically based on the number of nodes and GPUs per node.
    $$ \text{world\_size} = \text{num\_nodes} \times \text{num\_gpus\_per\_node} $$

2. **Master Address and Port**:
    - For a multi-node setting, we cannot use `localhost` as the `master_addr` as it won't be accessible to other nodes. You'd need to use the IP address of the node that acts as the master node.
    - The master port (`master_port`) can be left as is, but ensure the port is open and accessible from all nodes. Can use some program like `netcat` to check if the port is open
    before running the script.

3. **Node Rank Determination**:
    - We've included an argument `--node_rank`. For a truly distributed setting, we'd typically determine the node rank programmatically, perhaps from environment variables set by your scheduler (like PBS or SLURM). If we're manually setting the node rank, ensure each node has a unique rank.
