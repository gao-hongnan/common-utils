First, some context:

CUDA's memory allocator will attempt to find a block of unused memory that fits the size of the requested tensor. If it cannot find a large enough contiguous block of memory due to fragmentation, it will try to flush the cache and allocate again, thus leading to "allocation retries".

The concept of fragmentation is akin to what happens on disk drives: over time, as you allocate and deallocate memory, you're left with "holes" of free memory between used chunks. Even if the total free memory might be enough to satisfy a new allocation request, the free memory might not be contiguous.

Here's a conceptual illustration:

```
Memory: |XXXX|----|XXXX|------|XXXXX|---|XXXXX|
Legend:  X - used block, - - free block
```

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

def tensor_with_vram(vram_MB, dtype=torch.float32, device="cuda"):
    """Create a tensor that consumes the specified VRAM in MB."""

    # Compute number of elements based on desired VRAM and data type size
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    num_elements = int(vram_MB * 1e6 / bytes_per_element)

    # Create a 1D tensor with the computed number of elements
    tensor = torch.empty(num_elements, dtype=dtype, device=device)

    return tensor

# Test the function by creating a tensor that consumes 1GB of VRAM
tensor_1GB = tensor_with_vram(1024)  # 1024 MB = 1 GB

# Check tensor size and actual MB consumed
print(tensor_1GB.size(), tensor_1GB.element_size() * tensor_1GB.numel() / 1e6)



def get_memory_allocated():
    return torch.cuda.memory_allocated() / 1e6  # MB

def get_memory_cached():
    return torch.cuda.memory_reserved() / 1e6  # MB

print(f"Initial memory allocated: {get_memory_allocated()} MB")
print(f"Initial memory cached: {get_memory_cached()} MB")

# 1. Force Fragmentation
# Create tensors to induce fragmentation.

tensors = [tensor_with_vram(vram_MB=1024) for _ in range(39)]
print(f"Memory after tensor allocations: {get_memory_allocated()} MiB")



# Deallocate some tensors to induce fragmentation
for i in range(5, 35, 5):  # Removing tensors at intervals
    del tensors[i]
torch.cuda.synchronize()  # Ensure CUDA operations are synchronized


print(f"Memory after deleting some tensors: {get_memory_allocated()} MB")

# 2. Attempt to allocate a contiguous tensor
# Try to allocate a contiguous tensor of 2 GB
tensor_2gb = tensor_with_vram(1024 * 2, device="cuda").contiguous()
print(f"Memory after allocating a contiguous tensor: {get_memory_allocated()} MB")


# 2. Attempt Allocations
# We'll try to allocate tensors in a loop and catch any failures.
# success_count = 0
# failure_count = 0
# while success_count < 10:
#     try:
#         tensor = torch.randn(tensor_size).cuda()
#         success_count += 1
#     except RuntimeError as e:
#         failure_count += 1
#         if failure_count > 10:  # Limit to prevent infinite loop
#             break

# print(f"Successful allocations after fragmentation: {success_count}")
# print(f"Failed allocations after fragmentation: {failure_count}")

# 3. Memory Summary
memory_summary = torch.cuda.memory_summary().split('\n')
for line in memory_summary:
    if 'num_alloc_retries' in line:
        print(line)
```