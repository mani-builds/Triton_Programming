#!/usr/bin/env python

import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax(x_ptr, y_ptr, M, N,
                  stride_x, stride_y,
                  bs: tl.constexpr):
    # Launch a program for a single row and compute softmax for that row
    row_idx = tl.program_id(0)

    x_ptr += row_idx * stride_x
    y_ptr += row_idx * stride_y
    # compute softmax for this row
    log2_e = 1.44
    max = float('-inf')
    den = 0.0
    for start in range(0, N, bs):
        cols = start * bs + tl.arange(0, bs)
        mask = cols < N
        x_row = tl.load(x_ptr + cols, mask=mask, other=0.0)

        # max and rolling den
        #_max = tl.zeros([bs], dtype=tl.float32)
        #_den = tl.zeros([bs], dtype=tl.float32)
        row_max = tl.max(x_row, axis=0)
        max_new = tl.where(max > row_max, max, row_max)
        den_sum = tl.exp2((x_row - max_new) * log2_e)
        den = den * exp2((max - max_new) * log2_e)
        + tl.sum(den_sum, axis=0)  # rescale

        max = max_new

    # softmax
    for start in range(0, N, bs):
        cols = start * bs + tl.arange(0,bs)
        mask = cols < N
        x_row = tl.load(x_ptr + cols, mask=mask, other=0.0)
        exp_row = tl.exp2((x_row - max) * log2_e)
        soft = exp_row / den

        # HBM
        tl.store(y_ptr + cols, soft, mask = mask)

    return
