import gc
import os
import random
from enum import IntEnum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rich.pretty import pprint


class MemoryUnits(IntEnum):
    MB = int(1e6)
    GB = int(1e9)
    MiB = 2**20
    GiB = 2**30


def seed_all(seed: Optional[int] = 1992, seed_torch: bool = True) -> int:
    """
    Seed all random number generators.

    Parameters
    ----------
    seed : int, optional
        Seed number to be used, by default 1992.
    seed_torch : bool, optional
        Whether to seed PyTorch or not, by default True.
    """
    # fmt: off
    os.environ["PYTHONHASHSEED"] = str(seed)        # set PYTHONHASHSEED env var at fixed value
    np.random.seed(seed)                            # numpy pseudo-random generator
    random.seed(seed)                               # python's built-in pseudo-random generator

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)                # pytorch (both CPU and CUDA)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    # fmt: on
    return seed


def get_gpu_memory_info(
    device: Optional[torch.device] = None,
    as_dataframe: bool = False,
    unit: MemoryUnits = MemoryUnits.MB,
) -> Dict[str, float]:
    """
    Retrieves GPU memory information for a specified device using PyTorch.

    Parameters
    ----------
    device : torch.device, optional
        The device for which the memory information is required.
        Defaults to the current default CUDA device.
    unit : str, optional
        Desired memory unit: 'MB' or 'GB'. Defaults to 'MB'.

    Returns
    -------
    Dict[str, float]
        A dictionary containing memory details in the specified unit.

    Note: The 'free' memory is computed as 'reserved' - 'allocated'.
    """

    divisor = unit.value

    # Fetch memory details
    # restautant analogy
    allocated = torch.cuda.memory_allocated(device) / divisor  # allocated = in-use
    reserved = (
        torch.cuda.memory_reserved(device) / divisor
    )  # reserve = cache + allocated
    total = torch.cuda.get_device_properties(device).total_memory / divisor
    free_in_reserved = (
        reserved - allocated
    )  # This is an estimation https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch

    results: Dict[str, float] = {
        "allocated": allocated,
        "reserved": reserved,
        "free_in_reserved": free_in_reserved,
        "total": total,
    }

    if as_dataframe:
        return pd.DataFrame(results, index=[0])
    return results


def tensor_with_vram(
    vram: float, dtype=torch.float32, device="cuda", unit: MemoryUnits = MemoryUnits.MB
) -> torch.Tensor:
    """
    Create a tensor that consumes the specified VRAM in MB.

    Parameters
    ----------
    vram : float
        The desired amount of GPU memory (in megabytes/gigabytes) that the tensor should consume.
    dtype : torch.dtype, optional
        The data type of the tensor elements, by default torch.float32.
        - 4 bytes 32 bit
        - 2 bytes 16bit
    device : str, optional
        The device on which to allocate the tensor, by default "cuda".
    unit : str, optional
        Memory unit for vram: 'MB' or 'GB'. Defaults to 'MB'.

    Returns
    -------
    torch.Tensor
        A tensor allocated on the specified device consuming the desired VRAM.

    Example
    -------
    >>> tensor = tensor_with_vram(10.0, dtype=torch.float32)  # Create a tensor that roughly consumes 10MB of VRAM
    >>> results = get_gpu_memory_info(device="cuda", as_dataframe=True)
    >>> print(f"Memory after initial tensor allocations: {results['allocated']} MiB")
        Memory after initial tensor allocations: 10.0 MiB
    """
    divisor = unit.value

    # 1. bytes per element refers to how many bytes are required to store a
    #    single element of the tensor.
    #    For float32, it is 4 bytes.
    bytes_per_element = torch.tensor(
        [], dtype=dtype
    ).element_size()  # 4 bytes if dtype is float32

    # 2. Convert the desired VRAM from MB to bytes.
    #    1 MB = 1e6 bytes
    vram_in_bytes = int(vram * divisor)

    # 3. We can now compute the number of elements required to consume the
    #    desired VRAM. For example, if we want to consume 10MB of VRAM and
    #    each element is 4 bytes, then we need 2.5 million elements (10MB / 4 bytes).
    num_elements = int(vram_in_bytes / bytes_per_element)

    # Create a 1D tensor with the computed number of elements
    tensor = torch.empty(num_elements, dtype=dtype, device=device)
    assert tensor.size() == (num_elements,)

    return tensor


def fill_cuda_memory_with_tensors(
    vrams: Union[float, List[float]],
    unit: MemoryUnits = MemoryUnits.MB,
    contiguous: bool = False,
) -> List[torch.Tensor]:
    """
    Allocate tensors on the GPU based on specified sizes.

    Parameters:
    ----------
    vrams : Union[float, List[float]]
        If float (e.g., 10.0), will allocate N tensors each of vram size 10MB.
        If list of floats (e.g., [10.0, 20.0, 30.0]), will allocate tensors of
        vram sizes 10MB, 20MB, and 30MB respectively.
    unit : str, optional
        Memory unit for vram: 'MB' or 'GB'. Defaults to 'MB'.

    Returns:
    -------
    List[torch.Tensor]
        A list containing the allocated tensors.
    """

    tensors = []
    if isinstance(vrams, float):  # Allocate N blocks of the same size
        memory_full = False
        while not memory_full:
            try:
                if contiguous:
                    tensor = tensor_with_vram(vrams, unit=unit).contiguous()
                else:
                    tensor = tensor_with_vram(vrams, unit=unit)
                tensors.append(tensor)
            except RuntimeError:
                memory_full = True
                break
    elif isinstance(vrams, list):  # Allocate tensors with specified sizes
        for size in vrams:
            try:
                if contiguous:
                    tensor = tensor_with_vram(size, unit=unit).contiguous()
                else:
                    tensor = tensor_with_vram(size, unit=unit)
                tensors.append(tensor)
            except RuntimeError:
                print(
                    f"Failed to allocate tensor of size {size}{unit}. Stopping allocation."
                )
                break
    else:
        raise ValueError("tensor_sizes must be either a float or a list of floats.")

    return tensors


def delete_one_tensor_from_cuda_memory(tensor: torch.Tensor, empty_cache=True) -> None:
    del tensor

    if empty_cache:
        print("doing gc.collect before empty_cache")
        gc.collect()
        torch.cuda.empty_cache()


def delete_tensors_from_cuda_memory(
    tensors: List[torch.Tensor], num_to_delete: int, empty_cache=True
) -> List[torch.Tensor]:
    for _ in range(num_to_delete):
        index_to_delete = random.randint(0, len(tensors) - 1)
        tensors_to_delete = tensors.pop(index_to_delete)
        del tensors_to_delete

    if empty_cache:
        print("doing gc.collect before empty_cache")
        gc.collect()
        torch.cuda.empty_cache()
    return tensors


def induce_fragmentation_experiment(
    initial_vram: float = 4.0,
    vram_unit: MemoryUnits = MemoryUnits.MB,
    tensor_dtype: torch.dtype = torch.float32,
    big_tensor_vram: float = 2.0,
    big_tensor_unit: MemoryUnits = MemoryUnits.GiB,
    seed_value: Optional[int] = 412,
    seed_torch_flag: bool = True,
) -> None:
    """
    Runs a GPU memory experiment using PyTorch.

    Parameters:
    -----------
    initial_vram : float, optional
        Initial VRAM to allocate for tensor fill, default is 4.0.
    vram_unit : MemoryUnits, optional
        VRAM unit for the initial allocation, default is MemoryUnits.MB.
    tensor_dtype : torch.dtype, optional
        Data type for the tensors, default is torch.float32.
    big_tensor_vram : float, optional
        VRAM for the larger tensor to test allocation, default is 2.0.
    big_tensor_unit : MemoryUnits, optional
        VRAM unit for the big tensor, default is MemoryUnits.GiB.
    seed_value : int, optional
        Seed value for RNG, default is 412.
    seed_torch_flag : bool, optional
        Flag to seed PyTorch or not, default is True.

    Returns:
    --------
    None
    """

    seed_all(seed_value, seed_torch=seed_torch_flag)

    buffer = CustomAllocator(buffer=750000000 * 2) #3*2=6gb
    placeholder_optimizer = buffer.allocate(1000000000)
    placeholder_model = buffer.allocate(500000000)

    tensors = fill_cuda_memory_with_tensors(
        vrams=initial_vram, unit=vram_unit, contiguous=True
    )

    num_tensors = len(tensors)
    print(f"tensor length: {num_tensors}")

    results = get_gpu_memory_info(as_dataframe=False, unit=MemoryUnits.GiB)
    print("Memory after initial tensor allocations")
    pprint(results)
    pprint(torch.cuda.mem_get_info())

    # Randomly delete half of the tensors to induce fragmentation
    num_tensors_to_delete = min(len(tensors) // 2, num_tensors // 2)
    print(f"num tensors to delete: {num_tensors_to_delete}")

    delete_tensors_from_cuda_memory(
        tensors=tensors, num_to_delete=num_tensors_to_delete
    )

    torch.cuda.synchronize()  # Ensure CUDA operations are synchronized

    results = get_gpu_memory_info(as_dataframe=False, unit=MemoryUnits.GiB)
    print("Memory after deleting some tensors")
    pprint(results)
    pprint(torch.cuda.mem_get_info())

    # Try to allocate a tensor that's bigger than the gaps we've created but smaller than total free memory
    try:
        big_tensor = tensor_with_vram(
            vram=big_tensor_vram,
            dtype=tensor_dtype,
            device="cuda",
            unit=big_tensor_unit,
        ).contiguous()
        print("Successfully allocated big_tensor!")

    except RuntimeError as e:
        print(f"Allocation failed due to fragmentation: {e}")

    results = get_gpu_memory_info(as_dataframe=True)
    print(results)

    # Memory Summary
    memory_summary = torch.cuda.memory_summary()
    print(memory_summary)

    memory_stats = torch.cuda.memory_stats()
    print(f"Num Alloc Retries: {memory_stats['num_alloc_retries']}")

    # Getting cached memory
    cached_memory = (
        memory_stats["reserved_bytes.all.peak"]
        - memory_stats["allocated_bytes.all.peak"]
    )
    print(f"Cached Memory: {cached_memory / (1024 ** 3):.2f} GiB")


class SimpleModel(nn.Module):
    def __init__(self, hidden_dim: int = 100) -> None:
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def save_model_optimizer(
    model: nn.Module, optimizer: optim.Optimizer, path: str = "model_and_optimizer.pth"
) -> None:
    """
    Save model and optimizer to a given path.

    Parameters:
    ----------
    model : nn.Module
        PyTorch model to save.
    optimizer : optim.Optimizer
        Optimizer associated with the model to save.
    path : str, optional
        Destination file path. Default is "model_and_optimizer.pth".
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def memory_experiment_with_torch_save() -> None:
    """
    Perform a memory experiment with torch.save focused on model and optimizer.
    """

    # Step 1: Measure current GPU memory
    initial_memory = get_gpu_memory_info(as_dataframe=True, unit=MemoryUnits.GiB)
    print("Initial GPU Memory:")
    pprint(initial_memory)

    # Step 2: Create, save model and optimizer, and measure memory
    model = SimpleModel().cuda()  # Moving model to GPU
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    after_model_and_optimizer_creation_memory = get_gpu_memory_info(
        as_dataframe=True, unit=MemoryUnits.GiB
    )
    print("Memory after model and optimizer creation:")
    pprint(after_model_and_optimizer_creation_memory)

    # Ensure synchronization before memory check
    torch.cuda.synchronize()

    save_path = "model_and_optimizer.pth"
    # state_dict = {
    #     'model': model.state_dict(),
    #     'optimizer': optimizer.get_rng_state(),
    # }
    save_model_optimizer(model, optimizer, path=save_path)

    # Ensure synchronization before memory check
    torch.cuda.synchronize()

    after_save_memory = get_gpu_memory_info(as_dataframe=True, unit=MemoryUnits.GiB)
    print("Memory after model and optimizer creation and saving:")
    pprint(after_save_memory)

    # Cleanup the temporary file
    os.remove(save_path)

    # Print difference
    difference = (
        after_save_memory["allocated"][0]
        - after_model_and_optimizer_creation_memory["allocated"][0]
    )
    print(f"Memory Difference due to model and optimizer saving: {difference:.2f} GiB")


class CustomAllocator:
    def __init__(self, buffer):
        # num_elements = (buffer * 2**30) // 4  # Assuming float32 which is 4 bytes
        self.buffer = torch.empty(buffer, device="cuda")
        self.next_free_position = 0
        self.free_slices = []  # This will store start and end positions of free slices

    def allocate(self, num_elements):
        # First, try to find a suitable free slice
        for idx, (start, end) in enumerate(self.free_slices):
            if end - start >= num_elements:
                tensor_slice = self.buffer[start : start + num_elements]
                self.free_slices.pop(idx)
                return tensor_slice

        # If no free slice is found, allocate from the next free position
        tensor_slice = self.buffer[
            self.next_free_position : self.next_free_position + num_elements
        ]
        self.next_free_position += num_elements
        return tensor_slice

    def deallocate(self, tensor_slice):
        # Simply mark the slice as free
        start = tensor_slice.data_ptr() - self.buffer.data_ptr()
        end = start + tensor_slice.numel()
        self.free_slices.append((start, end))

# # Initialize the custom allocator with a buffer size
# buffer_size = 100000  # This is a small size just for the example
# allocator = CustomAllocator(buffer_size)

# # Allocate a tensor
# tensor1 = allocator.allocate(5000)
# print(f"Allocated tensor1: {tensor1.size()}")

# # Allocate another tensor
# tensor2 = allocator.allocate(15000)
# print(f"Allocated tensor2: {tensor2.size()}")

# # Deallocation
# allocator.deallocate(tensor1)

# # Let's allocate again and see if it reuses the deallocated space
# tensor3 = allocator.allocate(5000)
# print(f"Allocated tensor3: {tensor3.size()}")  # This should ideally use the space that tensor1 was using

# # Checking the pointer addresses to ensure reuse
# print(f"Tensor1 address: {tensor1.data_ptr()}")
# print(f"Tensor3 address: {tensor3.data_ptr()}")  # This should be same as tensor1's address if reuse was successful

