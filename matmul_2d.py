#!/usr/bin/env python

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def matmul_with_2d_grid(a_ptr, b_ptr, c_ptr, M, N, K,
                        stride_am, stride_ak, stride_bk,
                        stride_bn, stride_cm, stride_cn,
                        BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_N : tl.constexpr,
                        PHASE_SIZE_K: tl.constexpr):
    # map data to blocks
    # Each blocked program processes one block of Output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # offsets and mask (Boundary conditions)
    # to split M and N into blocks
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M #1D
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N #1D

    mask_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) < M
    mask_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) < N
    # K into phases
    offs_k = tl.arange(0, PHASE_SIZE_K)

    # 2-D pointers
    a_ptrs = a_ptr + offs_am[:, None]*stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    #Logic
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, PHASE_SIZE_K)):
        # check the bounds of K, since M and N are already checked
        mask_a = (mask_m[:, None]) & (offs_k[None, :] < K - k * PHASE_SIZE_K)
        mask_b = (offs_k[None, :] < K - k * PHASE_SIZE_K) & (mask_m[:, None])
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        accumulator += tl.dot(a, b)

        # increase offsets so next iteration loads next block
        a_ptrs += PHASE_SIZE_K * stride_ak
        b_ptrs += PHASE_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    #Write to HBM
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)

def matmul(a, b, bs = 32):
    assert a.shape[1] == b.shape[0], "Incompatible shapes"
    assert a.is_contiguous, "Non-contiguous"
    m,k = a.shape
    n = b.shape[1]
    c = torch.empty((m,n), device=DEVICE, dtype=torch.float16)
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    grid = (triton.cdiv(m, bs), triton.cdiv(n, bs))

    matmul_with_2d_grid[grid](a, b, c,
                        m, n, k,
                        stride_am, stride_ak, stride_bk,
                        stride_bn, stride_cm, stride_cn,
                        bs, bs, bs)

    return c


a = torch.ones((96, 96), dtype=torch.float16, device=DEVICE)
b = torch.ones((96, 96), dtype=torch.float16, device=DEVICE)

c = matmul(a,b)
print("output: ", c)
