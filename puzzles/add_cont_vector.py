import triton
import triton.language as tl
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_const_vector(x_ptr, z_ptr, c, N0, B0: tl.constexpr):
    # Map data to pid
    pid = tl.program_id(0)
    offsets = pid * B0 + tl.arange(0,B0)
    x_ptrs = x_ptr + offsets
    mask = offsets < N0
    # load data
    x = tl.load(x_ptrs, mask=mask)
    # logic
    z = x + c
    # store data to HBM
    z_ptrs = z_ptr + offsets
    tl.store(z_ptrs, z, mask=mask)

def add_const(x,c):
    assert x.device == DEVICE
    assert x.is_contiguous
    n = x.numel()
    z = torch.empty_like(x, device=DEVICE)
    b0 = n
    grid = (b0,)
    add_const_vector[grid](x,z,c,n,b0)
    return z

x = torch.ones((32,), device = DEVICE)
c = 10
z = add_const(x,c)
print("Input: ", x)
print("Output: ", z)
