#!/usr/bin/env python

import torch
import triton
import triton.language as tl

M = 96
K = 96
N = 96

BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def matmul_navie_kernel(a_ptr, b_ptr, c_ptr, M,N,K,
                        stride_am, stride_ak, stride_bk, stride_bn,
                        stride_cm, stride_cn,
                        BLOCK_SIZE_M:tl.constexpr, BLOCK_SIZE_N : tl.constexpr,
                        BLOCK_SIZE_K : tl.constexpr,
                        ACTIVATION : tl.constexpr):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    # Row-wise calulcation of blocks
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # get the staring pointers
    # &X[i,j] = X + i * stride_xi + j * stride_xj
    # a[m:m+BLOCK_M, k:k+BLOCK_K] = a_ptr + (m : m+BLOCK_M)[:,None] * A.stride(0)
    #  + (k : k+BLOCK_K)[None,:] * A.stride(1)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 2-D pointers
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None,:] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None,:] * stride_bn

    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K,BLOCK_SIZE_K)):
        # Load the next block of A and B by checking the bounds of k
        a = tl.load(a_ptrs, mask = offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask = offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # accumulate along the K dimension
        accumulator = tl.dot(a, b, accumulator)
        # advance the pointers to next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # Write back the block of output matrix C with masks
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask = c_mask)

# we can fuse matmul with activation by providing it as ACTIVATION meta-parameter to matmul kernel
@triton.jit
def leaky_relu(x):
    return tl.where(x>=0, x, 0.01 * x)

def matmul(a,b, activation=""):
    # check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dims"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32


    c = torch.empty((M,N), device=DEVICE, dtype = torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = (triton.cdiv(M, BLOCK_SIZE_M) *
                         triton.cdiv(N, BLOCK_SIZE_N),)
    # print("Grid ", grid)

    matmul_navie_kernel[grid](a,b,c,M,N,K,
                              a.stride(0), a.stride(1),
                              b.stride(0), b.stride(1),
                              c.stride(0), c.stride(1),
                              BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                              ACTIVATION = activation
                              )
    return c

#unit test
torch.manual_seed(0)
a = torch.rand((M,K), device=DEVICE, dtype=torch.float16) - 0.5
b = torch.rand((K,N), device=DEVICE, dtype=torch.float16) - 0.5

c_triton = matmul(a,b)
c_torch = torch.matmul(a,b)

assert torch.allclose(c_triton, c_torch, atol=1e-2, rtol=0), (c_triton, c_torch)

# c_triton_fused = matmul(a,b,"leaky_relu")
print("Matmul Output: ", c_triton)
print("Matmul with fused activation: ", matmul(a,b,"leaky_relu"))
