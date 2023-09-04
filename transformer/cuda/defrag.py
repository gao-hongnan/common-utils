import torch
import os

# Set the max_split_size_mb value
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'  # For example, 512MB

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