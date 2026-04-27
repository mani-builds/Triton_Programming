
# Sum of a batch of numbers.

# Uses one program blocks.
# Block size B0 represents a range of batches of x of length N0.
# Each element is of length T. Process it B1 < T elements at a time.

# zi=∑jT(xi,j)= for i=1…N0

import triton
import triton.language as tl
import torch

@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    row_idx = tl.program_id(0)

    sum = tl.zeros((), dtype=tl.float32) # scalar
    offset_col_start = row_idx * T + tl.arange(0,B1)
    # phases = int((T + B1 - 1) / B1)
    for i in range(0, tl.cdiv(T, B1).to(tl.int32)):
      mask_col = offset_col_start < (T * (row_idx+1))
      x_row = tl.load(x_ptr + offset_col_start, mask=mask_col)
      sum += tl.sum(x_row, axis=0)
      offset_col_start += B1

    tl.store(z_ptr + row_idx, sum, mask = row_idx < N0)
    return

def sum(x):
    N0, T = x.shape
    assert x.is_contiguous()
    z = torch.empty(size=(N0,), device = x.device)

    grid = (N0,)
    B0 = N0
    B1 = 32
    N1 = 32
    sum_kernel[grid](x, z, N0, N1, T, B0, B1)
    return z

x = torch.ones((4,200), device = torch.device(0))
print(sum(x))
