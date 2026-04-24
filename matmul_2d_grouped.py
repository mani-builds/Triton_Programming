#!/usr/bin/env python

import triton
import triton.language as tl
import torch

# "Super-grouping" or "Tiling of tiles" to increase L2 cache hit-rate

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def matmul_2d_grouped(a_ptr, b_ptr, c_ptr,M,N,K,
                      stride_am,stride_ak,
                      stride_bk,stride_bn,
                      stride_cm,stride_cn,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                      PHASE_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):

    # Map output block-data to programs
    # We need to change the odering of blocks so that it's no longer row-wise
    # So the usual pid_m and pid_n are changed to
    # (0,1,2,3,4,5)     -> ((0,3, ...)
    # (6,7,8,9,10,11)   -> (1,4, ...)
    # (12,13,14,15,16,17)->(2,5, ...)) # Here group_size of row is 3
    # This is called "Swizzling"
    pid_m, pid_n = tl.swizzle2d(tl.program_id(0), tl.program_id(1),
                                tl.num_programs(0), tl.num_programs(1),
                                GROUP_SIZE_M)
    # pid_m = tl.program_id(0)
    # pid_n = tl.program_id(1)

    # Offsets and poi nters
    #chunk along m/n/k dimensions
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) #1-D
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)#1-D
    rk = tl.arange(0, PHASE_SIZE_K)#1-D

    # 2-D offsets
    offs_am = rm[:,None] * stride_am + rk[None, :] * stride_ak
    offs_bn = rk[:, None] * stride_bk + rn[None, :] * stride_bn

    #2-D pointers
    a_ptrs = a_ptr + offs_am
    b_ptrs = b_ptr + offs_bn

    # Mask for M and N
    mask_m = (rm < M)[:, None] #2D
    mask_n = (rn < N)[None, :] #2D

    # Logic
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, PHASE_SIZE_K)):
        # mask for A and B to load blocks
        mask_a = mask_m & (rk < K - k*PHASE_SIZE_K)[None, :]#2D
        mask_b = (rk < K - k*PHASE_SIZE_K)[:, None] & mask_n

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        accumulator += tl.dot(a,b)

        # Move to next k-phase blocks of A and B
        a_ptrs += PHASE_SIZE_K * stride_ak
        b_ptrs += PHASE_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    # Write back to HBM
    offs_cm = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask_c = (rm < M)[:, None] & (rn < N)[None, :]
    c_ptrs = c_ptr + offs_cm

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

    matmul_2d_grouped[grid](a, b, c,
                        m, n, k,
                        stride_am, stride_ak, stride_bk,
                        stride_bn, stride_cm, stride_cn,
                        bs, bs, bs, GROUP_SIZE_M = 3)

    return c


a = torch.ones((100, 150), dtype=torch.float16, device=DEVICE)
b = torch.ones((150, 200), dtype=torch.float16, device=DEVICE)

c = matmul(a,b)
print("output: ", c)
