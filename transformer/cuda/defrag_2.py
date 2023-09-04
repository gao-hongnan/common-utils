import torch
import os
import torch
from typing import Dict, Optional
import pandas as pd


def tensor_with_vram(vram_MB, dtype=torch.float32, device="cuda"):
    """Create a tensor that consumes the specified VRAM in MB."""

    # Compute number of elements based on desired VRAM and data type size
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    num_elements = int(vram_MB * 1e6 / bytes_per_element)

    # Create a 1D tensor with the computed number of elements
    tensor = torch.empty(num_elements, dtype=dtype, device=device)

    return tensor


def get_memory_allocated():
    return torch.cuda.memory_allocated() / 1e6  # MB


def get_gpu_memory_info(
    device: Optional[torch.device] = None, as_dataframe: bool = False
) -> Dict[str, float]:
    """
    Retrieves GPU memory information for a specified device using PyTorch.

    Parameters
    ----------
    device (torch.device, optional): The device for which the memory information is required.
                                       Defaults to the current default CUDA device.

    Returns
    -------
    - Dict[str, float]: A dictionary containing memory details:
        - 'allocated': Memory allocated by PyTorch tensors (in MB).
        - 'reserved': Memory reserved by the PyTorch caching allocator (in MB).
        - 'free': Estimated free memory available for PyTorch (in MB).
        - 'total': Total GPU memory (in MB).

    Note: The 'free' memory is computed as 'reserved' - 'allocated'.
    """

    # Fetch memory details
    allocated = torch.cuda.memory_allocated(device) / 1e6
    reserved = torch.cuda.memory_reserved(device) / 1e6
    total = torch.cuda.get_device_properties(device).total_memory / 1e6
    free = total - reserved + allocated  # This is an estimation

    results: Dict[str, float] = {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total,
    }

    if as_dataframe:
        return pd.DataFrame(results, index=[0])
    return results


# Example usage
memory_info = get_gpu_memory_info()
print(memory_info)


# create multiple tensors of different sizes
tensor_1 = torch.randn((int(5 * 1024 * 256), 1024), device="cuda").contiguous()  # 5GB
tensor_2 = torch.randn((int(3 * 1024 * 256), 1024), device="cuda").contiguous()  # 3GB
tensor_3 = torch.randn((int(7 * 1024 * 256), 1024), device="cuda").contiguous()  # 7GB
tensor_4 = torch.randn((int(25 * 1024 * 256), 1024), device="cuda").contiguous()  # 5GB

print(f"Memory after initial tensor allocations: {get_memory_allocated()} MiB")

# delete some tensors to induce fragmentation
del tensor_1
del tensor_2

torch.cuda.synchronize()  # Ensure CUDA operations are synchronized

print(f"Memory after deleting some tensors: {get_memory_allocated()} MiB")

# Now, try to allocate a tensor larger than any individual gap but smaller than the sum of gaps.
try:
    big_tensor = torch.randn(
        (int(8 * 1024 * 256), 1024), device="cuda"
    ).contiguous()  # 6GB tensor, larger than any of the gaps
    print("Successfully allocated big_tensor!")
except RuntimeError as e:
    print(f"Allocation failed due to fragmentation: {e}")

# Memory Summary
memory_summary = torch.cuda.memory_summary().split("\n")
for line in memory_summary:
    print(line)
    if "num_alloc_retries" in line:
        print(line)

memory_stats = torch.cuda.memory_stats()
print(memory_stats)
