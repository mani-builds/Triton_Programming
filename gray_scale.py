#!/usr/bin/env python
import os
os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import torch
import triton
import triton.language as tl

from pathlib import Path
import matplotlib.pyplot as plt

import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io

DEVICE = triton.runtime.driver.active.get_active_torch_device()
# %%

@triton.jit
def gray_scale_kernel(x_ptr, out_ptr, h, w,
                      stride_cn, stride_hn, stride_wn,
                      stride_ho, stride_wo,
                      bs0: tl.constexpr, bs1: tl.constexpr):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)

    # 1-d vector offsets
    offs_w = pid_w * bs0 + tl.arange(0, bs0)
    offs_h = pid_h * bs1 + tl.arange(0, bs1)

    # 2-d matrix offsets
    # offs = w * offs_h[:, None] + offs_w[None, :]

    # Input pointers for R, G, B channels
    # Layout: channel * stride_c + row * stride_h + col * stride_w
    offs_r = (0 * stride_cn) + (offs_h[:, None] * stride_hn) + (offs_w[None, :] * stride_wn)
    offs_g = (1 * stride_cn) + (offs_h[:, None] * stride_hn) + (offs_w[None, :] * stride_wn)
    offs_b = (2 * stride_cn) + (offs_h[:, None] * stride_hn) + (offs_w[None, :] * stride_wn)
    # 2-d mask, doesn't must go beyond either axis, therefore 'logical and'
    mask = (offs_h < h)[:, None] & (offs_w < w)[None, :]

    r = tl.load(x_ptr + offs_r, mask = mask) # r is in channel-0
    g = tl.load(x_ptr + offs_g, mask = mask)# g is in channel-1
    b = tl.load(x_ptr + offs_b, mask = mask)# b is in channel-2

    out = 0.2989*r + 0.5870*g + 0.1140*b

    # if (pid_w == 0 & pid_h ==0):
        # print("out_ptrs ", (out_ptr + offs).shape)
    # write to HBM
    offs_out = (offs_h[:, None] * stride_ho) + (offs_w[None, :] * stride_wo)
    tl.store(out_ptr + offs_out, out, mask = mask)

# Helper function
def gray_scale(x, bs):
    c, h, w = x.shape
    gray = torch.empty((h,w), dtype=x.dtype, device=DEVICE)
    stride_ho, stride_wo = gray.stride()
    assert gray.device == x.device, "Both ip and out should be on the same device"
    # assert x.is_contiguous(), "Should be contingous"

    # Get strides: (stride for C, stride for H, stride for W)
    stride_cn, stride_hn, stride_wn = x.stride()

    # grid is 2-D here
    # having a grid function is useful when we benchmark and auto-tune kernels
    grid = lambda meta: (tl.cdiv(w, meta['bs0']), tl.cdiv(h, meta['bs1']))
    gray_scale_kernel[grid](
        x, gray, h, w,
        stride_cn, stride_hn, stride_wn,
        stride_ho, stride_wo,
        bs0=bs[0], bs1=bs[1]
    )
    return gray

# %%
path_img = Path('/home/mani/cuda/triton/cute-dog.jpg')

img = io.decode_image(path_img)
img = img.to(DEVICE)
c, h, w = img.shape
print("Shape of RGB image: ", img.shape)
gray_img = gray_scale(img, bs=(32,32)) #.to('cpu')
print("Shape of grayscale image: ", gray_img.shape)
print(gray_img)
#change dim of gray image from (h,w) to (1, h, w) tensor for jpeg conversion
gray_img = gray_img[None, :].to('cpu')
print("Reshaped grayscale image: ", gray_img.shape)
print(gray_img)
jpeg_bytes = io.encode_jpeg(gray_img)
print("Raw byets of grayscale image: ", jpeg_bytes)
if isinstance(jpeg_bytes, torch.Tensor):
    jpeg_bytes = jpeg_bytes.cpu().numpy().tobytes()
    with open('raw_dog.jpeg', 'wb') as f:
        f.write(jpeg_bytes)
    print(f"Finished writing the grayscale output")
