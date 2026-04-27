#!/usr/bin/env python

import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_kernel(x_ptr, y_ptr, M, N,
                  stride_x, stride_y,
                  bs: tl.constexpr):
    # Launch a program for a single row and compute softmax for that row
    row_idx = tl.program_id(0)

    x_ptr += row_idx * stride_x
    y_ptr += row_idx * stride_y
    # compute softmax for this row
    # log2_e = 1.44269504
    row_max = float('-inf')
    den = 0.0
    for start in range(0, tl.cdiv(N, bs)):
        cols = start * bs + tl.arange(0, bs)
        mask = cols < N
        x_row = tl.load(x_ptr + cols, mask=mask, other=-float('inf'))

        # max and rolling den
        block_max = tl.max(x_row, axis=0)
        old_max = row_max
        row_max = tl.where(block_max > row_max, block_max, row_max)

        den_sum = tl.exp((x_row - row_max))
        den = den * tl.exp((old_max - row_max)) + tl.sum(den_sum, axis=0)  # rescale

    # softmax
    for start in range(0, tl.cdiv(N, bs)):
        cols = start * bs + tl.arange(0,bs)
        mask = cols < N
        x_row = tl.load(x_ptr + cols, mask=mask, other=-float('inf'))
        exp_row = tl.exp((x_row - row_max))
        soft = exp_row / den

        # HBM
        tl.store(y_ptr + cols, soft, mask = mask)

    return

def fused_softmax(x):
    M, N = x.shape
    y = torch.empty_like(x, device=x.device)

    print("Strides: ", x.stride(0), y.stride(0))
    # m = tl.next_power_of_2(M)
    grid = (M,)
    fused_softmax_kernel[grid](x, y , M, N,
                               x.stride(0), y.stride(0), bs=32)
    return y

def naive_torch(x :torch.Tensor):
    """ Computer row-wise softmax of X"""
    #max
    # Read MN elements, write M elements
    x_max = x.max(dim=1)[0]
    # Read MN + M elements, write MN elements
    z = x - x_max[:, None]
    # Read MN, write MN
    numerator = torch.exp(z)
    # Read MN, write M
    denominator = numerator.sum(dim=1)
    # Read MN + M, write MN
    softmax = numerator / denominator[:,None]
    # In Total: Read 5MN + 2M and Wrote 3MN + 2M elements
    return softmax

torch.manual_seed(0)
M = 1823
N = 781
x = torch.rand((M,N), device=torch.device(0))
output_triton = fused_softmax(x)
output_torch = torch.softmax(x, axis=1)
assert torch.allclose(output_triton, output_torch), (output_triton, output_torch)
# print(output)
# print(f"Shape of output: ", output.shape)
