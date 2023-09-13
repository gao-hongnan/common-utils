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


induce_fragmentation_experiment()
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
