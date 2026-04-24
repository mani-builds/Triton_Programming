#!/usr/bin/env python

import torch
import triton
import triton.language as tl
# 'Super grouping' or 'Tiling of Tiles' to increase the hit-rate of L2 Cache

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_SIZE_M : tl.constexpr,
                  BLOCK_SIZE_K : tl.constexpr,
                  BLOCK_SIZE_N : tl.constexpr,
                  GROUP_SIZE_M : tl.constexpr,
                  ):
    #program_id
    pid = tl.program_id(0)
    #no. of programs in M direction
    num_pid_m = tl.cdiv(M,BLOCK_SIZE_M)
    #no. of programs in N direction
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    #number of programs in group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    #ID of group this program is in
    group_id = pid // num_pid_in_group
    # Row-Id of first program in this group
    first_pid_m = group_id * GROUP_SIZE_M
    # If num_pid_m isn't divisible by 'GROUP_SIZE_M' then the last group is smaller
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # Within the groups, programs are ordered in a column-major order
    # Row-id of the program in the *launch-grid*
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # Col-id of the program in the *launch-grid*
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Starting multi-dimentional pointers of A and B
    # &a[m:m+BLOCK_M, k:k+BLOCK_K] = a_ptr + (m : m+BLOCK_M)[:,None] * A.stride(0)
    #                                      + (k : k + BLOCK_K)[None, :] * A.stride(1)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M # % M helps with bounds checking
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_am[:,None] * stride_am + offs_k[None, :] * stride_ak #2D pointer
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    accumulator = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K,BLOCK_SIZE_K)):
        # Load the next block of A and B
        a = tl.load(a_ptrs, mask= offs_k[None, :] < K - k * BLOCK_SIZE_K, other = 0.0)
        # 2D pointers
        b = tl.load(b_ptrs, mask = offs_k[:, None] < K - k * BLOCK_SIZE_K, other = 0.0)
        # accumulate along K dim
        accumulator = tl.dot(a,b, accumulator)
        # Advance to next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    # Write back the block of the output matrix to c_ptrs
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None,:] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a,b):
    #check constrains
    assert a.shape[0] == b.shape[1]
    assert a.is_contiguous()

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M,N), device=DEVICE, dtype=torch.float16)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 64
    GROUP_SIZE_M = 6
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N,BLOCK_SIZE_N),)
    matmul_kernel[grid](a, b, c,
                        M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N,
                        GROUP_SIZE_M)
    return c

torch.manual_seed(0)
a = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
b = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")

if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
