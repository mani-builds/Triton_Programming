import triton
import triton.language as tl
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# z_ij = relu(x_i * y_j) for i = 1,...,B0 j = 1,...,B1

@triton.jit
def mul_vector(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr,
                     B1: tl.constexpr):
    # Map data to pid
    pid_col = tl.program_id(0)
    pid_row = tl.program_id(1)

    offsets_y = pid_row * B1 + tl.arange(0,B1)
    offsets_x = pid_col * B0 + tl.arange(0,B0)

    x_ptrs = x_ptr + offsets_x
    y_ptrs = y_ptr + offsets_y
    mask_y = offsets_y < N1
    mask_x = offsets_x < N0
    # load data
    x = tl.load(x_ptrs, mask=mask_x)
    y = tl.load(y_ptrs, mask=mask_y)
    # logic
    z = x[None, :] * y[:, None]
    z = tl.where(z > 0, z, 0.0)
    # store data to HBM
    offsets_z = offsets_y[:, None] * N0 + offsets_x[None,:]
    mask_z = mask_y[:, None] & mask_x[None, :]
    z_ptrs = z_ptr + offsets_z
    tl.store(z_ptrs, z, mask=mask_z)

def mul_const(x,y):
    assert x.device == DEVICE
    assert y.device == DEVICE
    assert y.is_contiguous
    assert x.is_contiguous
    n0 = x.numel()
    n1 = y.numel()

    z = torch.empty(size=(n1,n0), device=DEVICE)
    b0 = 32
    b1 = 32
    grid = (triton.cdiv(n1, b1),triton.cdiv(n0,b0))
    mul_vector[grid](x,y,z,n0,n1,b0,b1)
    return z

torch.manual_seed(123)
x = torch.randint(low = -10, high=10, size=(100,), device = DEVICE)
y = torch.randint(low = -10, high=10, size=(90,), device = DEVICE)
z = mul_const(x,y)
print("Input: ", x)
print("Input 2: ", y)
print("Output: ", z)
print("Output: ", z.shape)
