#!/usr/bin/env python

import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = driver.active.get_active_torch_device()

def is_hip():
    return driver.active.get_current_target().backend == "hip"

def is_cdna():
    return is_hip() and driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                    'gfx90a', 'gfx908')

def naive_torch(x :torch.Tensor):
    """ Computer row-wise softmax of X"""
    #max
    # Read MN elements, write M elements
    x_max = x.max(dim=1)[0]
    # Read MN + M elements, write MN elements
    z = x - x_max[:, None]
    # Read MN, write MN
    numerator = torch.exp(z)
    # Read MN, write M
    denominator = numerator.sum(dim=1)
    # Read MN + M, write MN
    softmax = numerator / denominator[:,None]
    # In Total: Read 5MN + 2M and Wrote 3MN + 2M elements
    return softmax

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE : tl.constexpr, num_stages: tl.constexpr):
    #starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages = num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        offsets = tl.arange(0, BLOCK_SIZE)
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = offsets < n_cols
        row = tl.load(row_start_ptr + offsets, mask=mask, other = -float('inf'))
        #diff for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        #Write back to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

#helper function
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = []

def softmax(x):
    n_rows, n_cols = x.shape

    #block size of each loop iteration is the smallest power of two greater than
    #the n_cols in 'x'
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    num_warps = 8

    # Number of software pipelining stages
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # preallocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy
    kernel = softmax_kernel.warmup(y,x, x.stride(0),y.stride(0),n_rows,n_cols,
                                   BLOCK_SIZE = BLOCK_SIZE, num_stages = num_stages,
                                   num_warps = num_warps, grid=(1,))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    if is_hip():
        # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
        # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
        # ISA SECTION (3.6.4 for CDNA3)
        # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
        # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
        # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
        # not required to be equal numbers of both types.
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
        # When we divide this number with WARP_SIZE we get maximum number of waves that can
        # execute on a CU (multi-processor)  in parallel.
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    #create a no. of persistent programs
    kernel[(num_programs, 1, 1)](y, x, x.stride(0),y.stride(0),n_rows,n_cols,
                                   BLOCK_SIZE, num_stages)
    return y


torch.manual_seed(0)
M = 1823
N = 781
x = torch.rand((M,N), device=DEVICE)
output_triton = softmax(x)
output_torch = torch.softmax(x, axis=1)
assert torch.allclose(output_triton, output_torch), (output_triton, output_torch)
# print(output)
# print(f"Shape of output: ", output.shape)
