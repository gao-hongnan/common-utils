First, some context:

CUDA's memory allocator will attempt to find a block of unused memory that fits the size of the requested tensor. If it cannot find a large enough contiguous block of memory due to fragmentation, it will try to flush the cache and allocate again, thus leading to "allocation retries".

The concept of fragmentation is akin to what happens on disk drives: over time, as you allocate and deallocate memory, you're left with "holes" of free memory between used chunks. Even if the total free memory might be enough to satisfy a new allocation request, the free memory might not be contiguous.

Here's a conceptual illustration:

```
Initial Memory (6GB): |XXXXXX| all contiguous
Free three non-contiguous blocks of 1GB each: |X-X-X-XX|
Allocate a model that is 2GB but requires contiguous memory: |XX| fails because there is no contiguous block of 2GB
Legend:  X - used block, - - free block
```

Consider

Even if the sum of the lengths of all free blocks (`-`) is greater than a new allocation request, the request might still fail if it requires a contiguous block larger than any individual free block.


## Memory Fragmentation and Contiguous Memory

Memory fragmentation and the need for contiguous memory are closely related concepts, and both influence the behavior of memory allocation retries. Let's delve deeper into the relationship:

### 1. Memory Fragmentation:

When you allocate and deallocate memory repeatedly, the free memory can get divided into multiple smaller chunks, rather than a single large block. This is known as memory fragmentation.

Suppose you allocate three tensors `A`, `B`, and `C` consecutively in memory, and then deallocate tensor `B`. Now, there's a "gap" in memory where `B` used to be. If you try to allocate a new tensor `D` that's larger than this gap but smaller than the combined size of all free memory chunks, the allocation will fail due to fragmentation, even though there's theoretically enough free memory.

### 2. Contiguous Memory:

Many GPU operations require tensors to be stored in contiguous blocks of memory for efficiency reasons. A tensor is contiguous if its elements are laid out in memory sequentially without any gaps. Due to operations like slicing, a tensor can become non-contiguous, meaning its elements are spread out in memory.

When you ask PyTorch to allocate memory for a tensor, it looks for a contiguous block of memory that can fit the entire tensor. If the available free memory is fragmented, PyTorch might not find a suitable block even if the total free memory is larger than the requested size.

### 3. Allocation Retries & Garbage Collection:

When PyTorch fails to allocate memory due to lack of a contiguous block, it doesn't give up immediately. Instead, it triggers Python's garbage collector to free up any tensors that are no longer referenced. After this cleanup, PyTorch retries the allocation. The hope is that the garbage collection will free up enough memory to create a larger contiguous block that can fit the requested tensor.

### Intuition:

Imagine a parking lot where cars (tensors) come and go, parking in available spaces. Over time, as some cars leave, there will be empty spaces scattered throughout the lot (fragmentation). Now, if a bus (a larger tensor) comes in, it needs a long contiguous space to park. Even if the combined empty spaces can fit multiple buses, the bus can't park unless there's a single long space available.

If the parking lot management can ask some cars to leave (garbage collection) and then rearrange the remaining cars to create a long empty space (contiguous block), the bus can park. This is analogous to PyTorch triggering garbage collection and then retrying memory allocation.

In summary, memory fragmentation can prevent the allocation of contiguous memory blocks, which are often needed for efficient GPU operations. Allocation retries, triggered by garbage collection, attempt to mitigate this issue by cleaning up and consolidating memory.

## Experiment 1

When you create and delete tensors on a CUDA device, the CUDA memory allocator used by PyTorch, which is built on top of cuDNN and other libraries, doesn't immediately release the freed memory back to the GPU. Instead, it keeps it in a pool so that it can quickly serve subsequent allocation requests without having to go through the potentially expensive process of requesting fresh GPU memory. This caching mechanism is particularly beneficial for deep learning workloads, where tensor allocations and deallocations can be frequent.

When you deleted some 1GB tensors, the associated memory was returned to this cache, but it was not immediately released back to the GPU's free memory pool. Thus, when you subsequently tried to allocate a 2GB tensor, the memory allocator found sufficient space in the cached memory to serve this request. The allocator can combine the cached blocks (previously occupied by the 1GB tensors you deleted) to create a contiguous 2GB block.

This behavior explains why you can still allocate a 2GB tensor even after fragmenting the memory by deleting several 1GB tensors. The caching mechanism allows the allocator to efficiently repurpose and coalesce freed blocks to serve subsequent allocations.

If you were to push the memory utilization to its limits and then induce fragmentation, you might observe allocation failures. However, in the scenario you presented, the allocator's caching mechanism is successfully mitigating the effects of fragmentation.

```python
import torch
import os

# Set the max_split_size_mb value
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'  # For example, 512MB

def get_memory_allocated():
    return torch.cuda.memory_allocated() / 1e6  # MB

# Allocate tensors of varying sizes to prevent efficient coalescing.
tensor_sizes = [(int(0.9 * 1024 * 256), 1024),  # Slightly less than 1GB
                (int(1.1 * 1024 * 256), 1024),  # Slightly more than 1GB
                (int(1 * 1024 * 256), 512),     # Different width, nearly 0.5GB
                (int(1 * 1024 * 256), 2048)]    # Different width, nearly 2GB

tensors = []

# Try to fill up the GPU memory with these tensors in a cyclic pattern.
memory_full = False
while not memory_full:
    for size in tensor_sizes:
        try:
            tensors.append(torch.randn(size, device="cuda"))
        except RuntimeError:
            memory_full = True
            break

print(f"Memory after initial tensor allocations: {get_memory_allocated()} MiB")

# Delete every third tensor and every fifth tensor to induce fragmentation.
# indices_to_delete = set(list(range(2, len(tensors), 3)) + list(range(4, len(tensors), 5)))

# for idx in reversed(sorted(indices_to_delete)):
#     del tensors[idx]
del tensors[0]
del tensors[10]
del tensors[20]
torch.cuda.synchronize()  # Ensure CUDA operations are synchronized

print(f"Memory after deleting some tensors: {get_memory_allocated()} MB")

# Now, try to allocate a tensor larger than any individual gap but smaller than the sum of gaps.
try:
    big_tensor = torch.randn((int(3.4 * 1024 * 256), 1024), device="cuda").contiguous()  # 3GB tensor
    print("Successfully allocated big_tensor!")
except RuntimeError as e:
    print(f"Allocation failed due to fragmentation: {e}")

# 3. Memory Summary
memory_summary = torch.cuda.memory_summary().split('\n')
for line in memory_summary:
    print(line)
    if 'num_alloc_retries' in line:
        print(line)

memory_stats = torch.cuda.memory_stats()
print(memory_stats)
```

## Num Alloc Retries and cudaMalloc

In PyTorch, the `"num_alloc_retries"` parameter refers to the number of times the allocator will attempt to allocate memory via `cudaMalloc` (see NVIDIA [docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356))
after encountering a failure (i.e., running out of memory). If a memory allocation via `cudaMalloc` fails, PyTorch will:

1. Flush its internal memory cache to free up any unused memory blocks.
2. Try the `cudaMalloc` call again.

This process helps PyTorch deal with memory fragmentation on the GPU. Memory fragmentation occurs when the GPU's free memory is divided into small chunks scattered throughout its memory space, rather than a contiguous block. Due to fragmentation, even if there is enough total free memory to satisfy a `cudaMalloc` request, the allocation might still fail because there isn't a single contiguous block of the required size.

By flushing its internal memory cache, PyTorch attempts to mitigate the impact of fragmentation by consolidating memory, potentially turning multiple smaller free chunks into a larger, contiguous block.

The `"num_alloc_retries"` parameter dictates how many times PyTorch will try this flush-and-retry process before giving up and raising an out-of-memory error. If all retries fail, then it's a strong indication that the GPU truly doesn't have enough free memory for the request, and it's not just a fragmentation issue.

## **Memory Fragmentation in GPU during Training: A Diagnosis**

**1. Log: CUDA Out-of-Memory (OOM) Error**

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 4.88 GiB (GPU 2; 39.59
GiB total capacity; 28.60 GiB already allocated; 3.92 GiB free; 33.95 GiB
reserved in total by PyTorch). If reserved memory is substantially greater than allocated memory, consider
setting max_split_size_mb to counteract fragmentation. Refer to PyTorch's Memory
Management and PYTORCH_CUDA_ALLOC_CONF documentation.
```

This error signifies that while the GPU might have some memory available, it may not be in a single contiguous block that matches the allocation request, suggesting potential memory fragmentation.

**2. Increasing `alloc_retries`**
`alloc_retries` progressively becoming higher, suggesting that there is a "continuous allocation and deallocation" process, causing "many gaps" in the gpu memory, consider the below simplified example:

**Illustrative Example:**
Imagine the GPU memory as a series of blocks. In a simplified scenario:
- **Initial State**: A fully occupied 6GB memory block: |XXXXXX|.
- **Deallocations**: After freeing up three non-contiguous blocks of 1GB each, the memory looks like: |X-X-X-XX|.
- **Allocation Attempt**: A subsequent effort to allocate a contiguous 2GB block (represented as |XX|) for a model would fail. Despite there being 3GB of free memory, no single contiguous space can accommodate the 2GB request.
- **Legend**: X - occupied block; - - free block.



## References and Further Readings

- https://fastai1.fast.ai/dev/gpu.html#unusable-gpu-ram-per-process
