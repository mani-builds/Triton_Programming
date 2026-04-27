
# Flashattn of a batch of numbers.

# A scalar version of FlashAttention.

# Uses zero programs. Block size `B0` represents `k` of length `N0`.
# Block size `B0` represents `q` of length `N0`. Block size `B0` represents `v` of length `N0`.
# Sequence length is `T`. Process it `B1 < T` elements at a time.

# $$z_{i} = \sum_{j} \text{softmax}(q_1 k_1, \ldots, q_T k_T)_j v_{j} \text{ for } i = 1\ldots N_0$$

# This can be done in 1 loop using a similar trick from the last puzzle.

import triton
import triton.language as tl
import torch

@triton.jit
def scalar_flash_attn_kernel(
    Q_ptr, K_ptr, V_ptr, Z_ptr,
    T, stride_qn, stride_kn, stride_vn, stride_zn,
    B1: tl.constexpr, N0: tl.constexpr
):
    # Program ID identifies which block of Q we are processing
    # (In this scalar puzzle, we assume 1 program handles N0 rows)
    row_idx = tl.arange(0, N0)

    # Initialize running statistics
    # m: running max, l: running sum, z: running weighted sum
    m_i = tl.full([N0], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([N0], dtype=tl.float32)
    z_i = tl.zeros([N0], dtype=tl.float32)

    # Load the fixed block of Q
    q = tl.load(Q_ptr + row_idx * stride_qn)

    # Loop over the sequence length T in steps of B1
    for start_j in range(0, T, B1):
        cols_idx = start_j + tl.arange(0, B1)

        # Load blocks of K and V
        k = tl.load(K_ptr + cols_idx * stride_kn)
        v = tl.load(V_ptr + cols_idx * stride_vn)

        # 1. Compute logits: qk_j
        # For scalar version, this is element-wise multiplication
        # (assuming 1D vectors for the puzzle logic)
        qk = q[:, None] * k[None, :] # Result is [N0, B1]

        # 2. Update running max
        m_curr = tl.max(qk, axis=1)
        m_next = tl.maximum(m_i, m_curr)

        # 3. Compute exponentials with rescaling
        # Alpha and Beta are the correction factors
        alpha = tl.exp(m_i - m_next)
        p = tl.exp(qk - m_next[:, None])

        # 4. Update running sum (denominator)
        l_curr = tl.sum(p, axis=1)
        l_next = (l_i * alpha) + l_curr

        # 5. Update running weighted sum (numerator)
        # We rescale the old z_i to the new max
        p_v = tl.sum(p * v[None, :], axis=1)
        z_i = (z_i * alpha) + p_v

        # Update stats for next iteration
        m_i = m_next
        l_i = l_next

    # Final normalization: divide by the total sum of exponentials
    z_final = z_i / l_i
    tl.store(Z_ptr + row_idx * stride_zn, z_final)

def flashattn(q, k ,v):
    T = q.numel()
    assert q.is_contiguous()
    z = torch.empty(size=(T,T), device = q.device)

    N0 = T
    B0 = 32
    grid = (triton.cdiv(T, B0),)
    flashattn_kernel[grid](q,k,v, z, N0, T, B0)
    return z

torch.random.manual_seed(123)
q = torch.randint(low = 0, high = 10, size=(4,), device = torch.device(0))
k = torch.randint(low = 0, high = 10, size=(4,), device = torch.device(0))
v = torch.randint(low = 0, high = 10, size=(4,), device = torch.device(0))
print("Input q: ", q)
print("Input k: ", k)
print("Input v: ", v)
print("Output: ", flashattn(q,k,v))

# def flashattn_spec(x):
#     x_max = x.max(1, keepdim=True)[0]
#     x = x - x_max
#     x_exp = x.exp()
#     return x_exp / x_exp.sum(1, keepdim=True)

# print("Torch output: ",flashattn_spec(x))
