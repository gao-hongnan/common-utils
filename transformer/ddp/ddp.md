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
