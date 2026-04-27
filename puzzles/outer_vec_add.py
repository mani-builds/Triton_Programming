import triton
import triton.language as tl
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# z_ij = x_i + y_j for i = 1,...,B0 j = 1,...,B1

@triton.jit
def add_vector(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr,
                     B1: tl.constexpr):
    # Map data to pid
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(0)
    offsets_0 = pid_0 * B0 + tl.arange(0,B0)
    offsets_1 = pid_1 * B1 + tl.arange(0,B1)
    x_ptrs = x_ptr + offsets_0
    y_ptrs = y_ptr + offsets_1
    mask_0 = offsets_0 < N0
    mask_1 = offsets_1 < N1
    # load data
    x = tl.load(x_ptrs, mask=mask_0)
    y = tl.load(y_ptrs, mask=mask_1)
    # logic
    z = x[None, :] + y[:, None]
    # store data to HBM
    offsets_z = offsets_1[:, None] * N0 + offsets_0[None,:]
    mask_z = mask_1[:, None] & mask_0[None, :]
    z_ptrs = z_ptr + offsets_z
    tl.store(z_ptrs, z, mask=mask_z)

def add_const(x,y):
    assert x.device == DEVICE
    assert y.device == DEVICE
    assert y.is_contiguous
    assert x.is_contiguous
    n0 = x.numel()
    n1 = y.numel()

    z = torch.empty_like(x, device=DEVICE)
    b0 = n0
    b1 = n1
    grid = (b0,b1)
    add_vector[grid](x,y,z,n0,n1,b0,b1)
    return z

x = torch.ones((32,), device = DEVICE)
y = torch.ones((32,), device = DEVICE)
z = add_const(x,y)
print("Input: ", x)
print("Input 2: ", y)
print("Output: ", z)
