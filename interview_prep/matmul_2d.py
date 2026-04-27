#!/usr/bin/env python
import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  bs_M: tl.constexpr, bs_N : tl.constexpr,
                  ps_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    # Map program IDs to data
    # "Super Grouping" or "Tiling of Tiles"
    pid_m, pid_n = tl.swizzle2d(tl.program_id(0), tl.program_id(1),
                                tl.num_programs(0), tl.num_programs(1),
                                GROUP_SIZE_M)
    # offsets and mask/ Load data
    # chunk along m/n/k dimensions
    rm = pid_m * bs_M + tl.arange(0, bs_M) #1D
    rn = pid_n * bs_N + tl.arange(0, bs_N)
    # rk = tl.arange(0, ps_K)

    #2D offsets
    # offs_a = rm[:, None] * stride_am + rk[None, :] * stride_ak
    # offs_b = rk[None, :] * stride_bk + rn[None, :] * stride_bn

    # matmul
    accumulator = tl.zeros((bs_M, bs_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, ps_K)):
        # K indices for this iteration
        current_rk = k * ps_K + tl.arange(0, ps_K)
        #mask
        mask_a = (rm < M)[:, None] & (current_rk < K )[None, :]
        mask_b = (current_rk < K)[:, None] & (rn < N)[None, :]

        #tiles of A and B
        a_tile_ptr = a_ptr + rm[:, None] * stride_am + current_rk[None, :] * stride_ak
        b_tile_ptr = b_ptr + current_rk[:, None] * stride_bk + rn[None, :] * stride_bn
        # load a and b
        a = tl.load(a_tile_ptr, mask=mask_a, other=0.0)
        b = tl.load(b_tile_ptr, mask=mask_b, other=0.0)
        # dot
        accumulator += tl.dot(a,b)
        # Move to next k-phase of A and B
        # offs_a += ps_K * stride_ak
        # offs_b += ps_K * stride_bk
    c = accumulator.to(tl.float32)
    # store to HBM
    offs_c = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask_c = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(c_ptr + offs_c, c, mask=mask_c)

def matmul(a, b, bs = 32):
    assert a.shape[1] == b.shape[0], "Incompatible shapes"
    assert a.is_contiguous(), "Non-contiguous"
    m,k = a.shape
    n = b.shape[1]
    c = torch.empty((m,n), device=DEVICE, dtype=torch.float16)
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    grid = (triton.cdiv(m, bs), triton.cdiv(n, bs))

    matmul_kernel[grid](a, b, c,
                        m, n, k,
                        stride_am, stride_ak, stride_bk,
                        stride_bn, stride_cm, stride_cn,
                        bs, bs, bs, GROUP_SIZE_M = 3)

    return c

DEVICE = torch.device(0)
a = torch.ones((100, 150), dtype=torch.float16, device=DEVICE)
b = torch.ones((150, 200), dtype=torch.float16, device=DEVICE)

c = matmul(a,b)
c_torch = torch.matmul(a,b)
assert torch.allclose(c, c_torch, atol=1e-2, rtol=1e-2), (c,c_torch)
# print(c)
