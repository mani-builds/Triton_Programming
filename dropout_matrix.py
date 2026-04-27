import triton
import tabulate
import torch
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Kernel to operate dropout over a matrix and use a vector of seeds - one per row.
@triton.jit
def dropout_for_matrix(x_ptr, out_ptr, M, N, seed_vec_ptr, p,
                      # stride_xm : tl.constexpr, stride_xn : tl.constexpr,
                      # stride_om : tl.constexpr, stride_on : tl.constexpr,
                      BLOCK_SIZE_N: tl.constexpr):
    row_idx = tl.program_id(0) # row idx for blocks, total should be M
    col_block_idx = tl.program_id(1) # col-block idx for blocks, could be N / BLOCK_SIZE_N

    #Dropout row-wise
    offsets = col_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) + row_idx * N
    mask = offsets < (row_idx + 1) * N

    # Use each one idx of seed for one row
    seed_ptr = seed_vec_ptr + row_idx
    # seed_mask = seed_ptr < M
    row_seed = tl.load(seed_ptr, mask = row_idx < M)
    random = tl.rand(row_seed, tl.arange(0,1)) # Generate one random no.
                                               # for the entire row
    row_keep = random > p

    row = tl.load(x_ptr + offsets, mask=mask)
    #Logic
    out_row = tl.where(row_keep, row / (1-p), 0.0)
    #HBM
    out_ptrs = out_ptr + offsets
    tl.store(out_ptrs, out_row, mask=mask)

def dropout(x, seed_vec, p):
    M, N = x.shape
    assert x.is_contiguous, "Non-contiguous"
    assert x.device == DEVICE
    out = torch.empty_like(x)
    #2-D grid
    # No. of rows of programs should be M
    grid = (M, triton.cdiv(N, 1024))
    dropout_for_matrix[grid](x, out, M, N, seed_vec, p, BLOCK_SIZE_N = 1024)
    return out

torch.random.manual_seed(123)
x = torch.rand((400,300), device=DEVICE)
m, n = x.shape
p = 0.5 # dropout rate
seed_vec = (torch.rand((m,), device=DEVICE) > p).to(torch.int32)

out = dropout(x, seed_vec, p)

m = torch.nn.Dropout(p=p)
out_torch = m(x).to(torch.float32)

print("Input: ", x)
print("Output: ", out)
print("Output Torch: ", out_torch)
# print(tabulate.tabulate([
#     ["input"] + x.tolist(),
#     ["output"] + out.tolist(),
# ]))
