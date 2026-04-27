#!/usr/bin/env python
import tabulate
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def _dropout(x_ptr, out_ptr, x_keep_ptr, # mask of 0s and 1s
             n_elements, p, bs: tl.constexpr):
    # Map data to blocks
    pid = tl.program_id(0)
    # Offsets
    offsets = pid * bs + tl.arange(0, bs)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # Logic
    out = tl.where(x_keep, x / (1-p), 0.0)
    # HBM
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def _seeded_dropout(x_ptr, out_ptr, n_elements, p, seed: tl.int32, bs:tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * bs + tl.arange(0,bs)
    mask = offsets < n_elements
    #randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random < p
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.where(x_keep, x/ 1-p, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)

def dropout(x, x_keep, p):
    assert x.is_contiguous(), "Non-contiguous"
    assert x.device == DEVICE, "Not same device"
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, 1024),)
    _dropout[grid](x, out, x_keep, n_elements, p, bs=1024)
    return out

def seeded_dropout(x, seed:torch.int32, p):
    assert x.is_contiguous(), "Non-contiguous"
    assert x.device == DEVICE, "Not same device"
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, 1024),)
    _seeded_dropout[grid](x, out, n_elements, p, seed, bs=1024)
    return out

size = 10
x = torch.rand(size=(size,), device=DEVICE)
p = 0.5 #Dropout max
x_keep = (torch.rand(size=(size,), device=DEVICE) > p).to(torch.int32)
output = dropout(x,x_keep,p)
output1 = seeded_dropout(x, seed=123, p=p)
output2 = seeded_dropout(x, seed=123, p=p)
output3 = seeded_dropout(x, seed=512, p=p)
print(tabulate.tabulate([
    ["input"] + x.tolist(),
    ["keep mask"] + x_keep.tolist(),
    ["output"] + output.tolist(),
]))
print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output1.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))
