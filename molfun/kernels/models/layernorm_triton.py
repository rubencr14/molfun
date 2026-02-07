# molfun/kernels/layernorm_triton.py
#
# Triton LayerNorm (inference) over last dimension:
#   y = (x - mean) / sqrt(var + eps) * gamma + beta
#
# Shapes:
#   x:     [M, D]   (M = tokens flattened, D = hidden size)
#   gamma: [D]
#   beta:  [D]
#   y:     [M, D]
#
# Notes:
# - fp32 accumulation for mean/var
# - outputs in same dtype as x (fp16/bf16)
# - inference only

import torch
import triton
import triton.language as tl

"""
This kernel implements standard Layer Normalization on a per-token basis (row-wise). 
It follows the formula y = (x - mean) / sqrt(var + eps) * gamma + beta. 
The calculation is performed in two passes: the first pass computes the mean and variance 
using the moments formula Var(X) = E[X^2] - (E[X])^2, accumulating in float32 for numerical 
stability. The second pass normalizes the input using the reciprocal square root (rsqrt) 
of the variance and applies the learned affine transformation parameters (gamma and beta).
"""


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 256}, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_warps=4),
        triton.Config({"BLOCK_D": 1024}, num_warps=8),
        triton.Config({"BLOCK_D": 2048}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def layernorm_fwd_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yd: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program per row (token)
    pid = tl.program_id(axis=0)
    row = pid

    # We process D in blocks of BLOCK_D (loop if D > BLOCK_D)
    # Accumulate mean/var in fp32
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    for d0 in range(0, D, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        mask = cols < D

        x = tl.load(
            x_ptr + row * stride_xm + cols * stride_xd,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / D
    var = sum_x2 / D - mean * mean
    rstd = tl.math.rsqrt(var + eps)

    # Second pass: normalize + affine and store
    for d0 in range(0, D, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        mask = cols < D

        x = tl.load(
            x_ptr + row * stride_xm + cols * stride_xd,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * rstd
        y = y * gamma + beta

        tl.store(
            y_ptr + row * stride_ym + cols * stride_yd,
            y.to(tl.float16) if tl.constexpr(x_ptr) is None else y.to(tl.float16),  # unused guard
            mask=mask,
        )


def layernorm_triton(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    LayerNorm over last dimension (inference).
    x:     [*, D]
    gamma: [D]
    beta:  [D]
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert gamma.dtype == x.dtype and beta.dtype == x.dtype
    assert gamma.ndim == 1 and beta.ndim == 1
    assert x.shape[-1] == gamma.shape[0] == beta.shape[0]

    # Flatten to [M, D]
    x2 = x.contiguous().view(-1, x.shape[-1])
    M, D = x2.shape

    y2 = torch.empty_like(x2)

    grid = (M,)
    layernorm_fwd_kernel[grid](
        x2, gamma, beta, y2,
        M=M, D=D,
        stride_xm=x2.stride(0), stride_xd=x2.stride(1),
        stride_ym=y2.stride(0), stride_yd=y2.stride(1),
        eps=eps,
    )

    return y2.view(*x.shape)
