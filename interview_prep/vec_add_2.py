import triton
import triton.language as tl
import torch

@triton.jit
def vec_outer_sum_kernel(a_ptr, b_ptr, c_ptr, N, bs: tl.constexpr):
    # Map output data to blocks / Programs
    pid_x = tl.program_id(0) # controls rows from A
    pid_y = tl.program_id(1) # controls cols from B

    # offsets and mask, load data
    offsets_a = pid_x * bs + tl.arange(0, bs)
    offsets_b = pid_y * bs + tl.arange(0, bs)
    mask_a = offsets_a < N
    mask_b = offsets_b < N

    a = tl.load(a_ptr + offsets_a, mask=mask_a) # mask for boundary conditions
    b = tl.load(b_ptr + offsets_b, mask=mask_b) # stored in SRAM or shared mem
    # logic
    c = a[:, None] + b[None, :]
    # write to HBM
    offsets_c = offsets_a[:, None] * N + offsets_b[None, :] # 2D
    mask_c = (offsets_a[:, None] < N) & (offsets_b[None, :] < N)
    tl.store(c_ptr + offsets_c, c, mask=mask_c)

    return

# helper func
def vec_outer_sum(a,b):
    assert a.is_contiguous() # It's do with how memory is handle by Triton
    assert b.is_contiguous() # locality helps with memory coalescing
    assert a.device == b.device

    N = a.numel()
    c = torch.empty(size=(N,N), device = a.device)
    B0 = 16
    grid = (triton.cdiv(N, B0), triton.cdiv(N, B0))
    vec_outer_sum_kernel[grid](a,b,c,N,B0)
    return c

x = torch.ones((40,), device=torch.device(0))
y = torch.ones((40,), device=torch.device(0))

print("x: ", x)
print("y: ", y)
z = vec_outer_sum(x,y)
print(z)
