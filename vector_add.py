#!/usr/bin/env python

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr,):
    pid = tl.program_id(axis=0)
    block_size = pid * BLOCK_SIZE
    offsets = block_size + tl.arange(0, BLOCK_SIZE)
    # mask to gaurd against out-of-bounds accesses
    mask = offsets < n_elements
    # multiple of blocksizes
    x = tl.load(x_ptr+offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    #write back to DRAM
    tl.store(output_ptr+offsets, output, mask=mask)

def add(x : torch.Tensor, y: torch.Tensor):
    # Preallocate the output
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x,y)
print(output_torch)
print(output_triton)
print(f"Max difference in outputs: ", f"{torch.max(torch.abs(output_torch - output_triton))}")
