
# Softmax of a batch of numbers.

# Uses one program blocks.
# Block size B0 represents a range of batches of x of length N0.
# Each element is of length T. Process it B1 < T elements at a time.

# zi=∑jT(xi,j)= for i=1…N0

import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    row_idx = tl.program_id(0)
    log2_e = 1.44269504
    row_start_ptr = x_ptr + row_idx * T

    # -- LOOP 1: Online Max and Sum --
    m_i = float('-inf') # Running Max
    l_i = 0.0          # Running Sum (Denominator)

    for start in range(0, T, B1):
        offsets = start + tl.arange(0, B1)
        mask = offsets < T
        x_chunk = tl.load(row_start_ptr + offsets, mask=mask, other=float('-inf'))

        # New local max
        m_next = tl.max(x_chunk, axis=0)
        m_new = tl.maximum(m_i, m_next)

        # Rescale the previous sum to the new max
        # If m_new > m_i, this shrinks the old sum
        l_i = l_i * tl.exp2((m_i - m_new) * log2_e)

        # Add the contribution of the current chunk
        p_chunk = tl.exp2((x_chunk - m_new) * log2_e)
        l_i += tl.sum(tl.where(mask, p_chunk, 0.0), axis=0)

        # Update running max
        m_i = m_new

    # -- LOOP 2: Store --
    z_row_start_ptr = z_ptr + row_idx * T
    for start in range(0, T, B1):
        offsets = start + tl.arange(0, B1)
        mask = offsets < T
        x_chunk = tl.load(row_start_ptr + offsets, mask=mask, other=float('-inf'))

        # Use the final global max (m_i) and global sum (l_i)
        num = tl.exp2((x_chunk - m_i) * log2_e)
        softmax_chunk = num / l_i

        tl.store(z_row_start_ptr + offsets, softmax_chunk, mask=mask)


def softmax(x):
    N0, T = x.shape
    assert x.is_contiguous()
    z = torch.empty(size=(N0,T), device = x.device)

    grid = (N0,)
    B0 = N0
    B1 = 32
    N1 = 32
    softmax_kernel[grid](x, z, N0, N1, T, B0, B1)
    return z

x = torch.randint(low = -10, high = 10, size=(4,2), device = torch.device(0))
print("Input: ", x)
print("Output: ", softmax(x))

def softmax_spec(x):
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)

print("Torch output: ",softmax_spec(x))
