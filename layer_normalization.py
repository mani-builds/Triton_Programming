#!/usr/bin/env python
import triton
import triton.language as tl
import torch

# Layer norm, x is a vector/matrix and y is the output vector/matrix
# 𝑦 = (𝑥−E⁡[𝑥] / √Var⁡(𝑥)+𝜖) ∗ 𝑤 + 𝑏

@triton.jit
def _layer_norm_fwd_fused(
        X, # pointer to input
        Y, # ptr to output
        W, # ptr to weights, same for all rows (1D vector)
        B, # ptr to bias, same for all rows (1D vector)
        Mean,
        Rstd,
        stride, # how much to increase the ptr when moving by 1 row
        N, # elements in a Row or No. of cols in X
        eps,
        bs: tl.constexpr
):
    # Map each row to a program
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    # mean
    mean = 0
    _mean = tl.zeros([bs], dtype=tl.float32)
    for start in range(0, N, bs):
        cols = start + tl.arange(0, bs)
        x = tl.load(X + cols, mask = cols < N, other = 0.0).to(tl.float32)
        _mean += x
    mean = tl.sum(_mean, axis=0) / N

    # variance
    var = 0
    _var = tl.zeros([bs], dytpe=tl.float32)
    for start in range(0, N, bs):
        cols = start + tl.arange(0,bs)
        a = tl.load(X + cols, mask = cols < N, other = 0.0)
        a = tl.where(cols < N, a - mean, 0)
        _var += a * a
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, var)

    # divison
    for start in range(0, N, bs):
        cols = start * bs + tl.arange(0, bs)
        x = tl.load(X + cols, mask = cols < N, other =0.0)
        p1 = (x - mean) * rstd
        w = tl.load(W + cols, mask = cols < N)
        b = tl.load(B + cols, mask=cols < N)
        p2 = p1 * w + b
        tl.store(Y + cols, p2, mask = cols < N) # store to HBM block-wise

