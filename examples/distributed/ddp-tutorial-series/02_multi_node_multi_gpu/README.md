# Multi Node Multi GPU

## Rendezvous vs Master?

The `master_addr` and `master_port` are indeed used in distributed training for specifying where the initial communication (rendezvous) happens. This is especially true for the `static` rendezvous backend in PyTorch's distributed training.

To give intuition:

1. **Static Rendezvous Backend**:

   The `static` backend, as the name suggests, means the addresses and ports are predetermined. It's the more traditional way to set up distributed training, where one explicitly specifies the master node's address and port. This is where processes will rendezvous to set up their communication.

   - If you use the `static` backend, you need to specify the `master_addr` and `master_port` unless a `rdzv_endpoint` is provided.
   - For instance, if you're initiating a distributed process group with `torch.distributed`, you might see something like:
     ```python
     torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://192.168.1.1:12345',  # where 192.168.1.1 is master_addr and 12345 is master_port
        rank=rank,
        world_size=world_size
     )
     ```

2. **Dynamic Rendezvous Backend**:

   Some modern approaches to distributed training utilize dynamic rendezvous methods. Instead of hardcoding `master_addr` and `master_port`, the system might discover available nodes dynamically and negotiate who does what. This is handy for cloud environments or kubernetes setups where you might not always have a fixed IP and port.

   - If you're using a dynamic rendezvous backend like `etcd`, `c10d`, etc., the setup might look different. In such cases, `rdzv_endpoint` might be provided to point to the service handling the rendezvous.

For clarity, when using the `static` backend:

- If `rdzv_endpoint` is **not** specified, `master_addr` and `master_port` are used.
- If `rdzv_endpoint` is specified, it takes precedence over `master_addr` and `master_port`.

When you're setting up distributed training, ensure you're consistent in the choice of your rendezvous method and the corresponding parameters to avoid confusion or unexpected behaviors.