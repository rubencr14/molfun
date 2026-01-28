# src/kernels/fused_linear_bias_residual_triton.py
#
# Fused: Y = residual + (H @ W^T + b)
# - H shape: [M, K]    (e.g., GELU output, flattened tokens)
# - W shape: [N, K]    (nn.Linear weight: [out_features, in_features])
# - b shape: [N]
# - residual shape: [M, N]   (the skip connection you add back)
# - Y shape: [M, N]
#
# In ESM-like MLP block, this is the "second projection + residual add" fusion.

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BM": 64,  "BN": 64,  "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 128, "BN": 64,  "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64,  "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64,  "BN": 64,  "BK": 64}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_linear_bias_residual_kernel(
    h_ptr,                       # H pointer (input activations) [M, K]
    w_ptr,                       # W pointer (weights) [N, K]
    b_ptr,                       # bias pointer [N]
    r_ptr,                       # residual pointer [M, N]
    y_ptr,                       # output pointer [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_hm: tl.constexpr,
    stride_hk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_rm: tl.constexpr,
    stride_rn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    IN_DTYPE: tl.constexpr,      # tl.float16 or tl.bfloat16 (match inputs)
):
    # 2D program ids -> one output tile [BM, BN]
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Loop over K in BK chunks: acc += H_tile @ W_tile^T
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)

        h_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        h = tl.load(
            h_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk,
            mask=h_mask,
            other=0.0,
        ).to(IN_DTYPE)  # keep in fp16/bf16 in registers

        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_nbk = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=w_mask,
            other=0.0,
        ).to(IN_DTYPE)

        w = tl.trans(w_nbk)  # [BK, BN]
        acc += tl.dot(h, w)  # fp32 accumulate

    # Add bias (broadcast across BM)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    z = acc + b[None, :]  # fp32

    # Load residual tile and add
    r_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    r = tl.load(
        r_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn,
        mask=r_mask,
        other=0.0,
    ).to(tl.float32)

    out_f32 = z + r

    # Cast back to input dtype and store
    out = out_f32.to(IN_DTYPE)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        out,
        mask=r_mask,
    )


def fused_linear_bias_residual_triton(
    h: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """
    Compute: out = residual + (h @ weight^T + bias) in one Triton kernel.

    Requirements (inference):
    - h:        [*, K]
    - weight:   [N, K]   (nn.Linear weight)
    - bias:     [N]
    - residual: [*, N]   (same leading dims as h, last dim = N)
    - dtype: fp16 or bf16, all on CUDA
    """
    assert h.is_cuda and weight.is_cuda and bias.is_cuda and residual.is_cuda, "All tensors must be on CUDA"
    assert h.dtype in (torch.float16, torch.bfloat16), "Use fp16/bf16"
    assert weight.dtype == h.dtype and bias.dtype == h.dtype and residual.dtype == h.dtype, "All dtypes must match"
    assert weight.ndim == 2 and bias.ndim == 1
    assert h.shape[-1] == weight.shape[1], "h in_features must match weight in_features (K)"
    assert weight.shape[0] == bias.shape[0], "weight out_features (N) must match bias"
    assert residual.shape[:-1] == h.shape[:-1], "residual leading dims must match h"
    assert residual.shape[-1] == weight.shape[0], "residual last dim must be N"

    # Flatten to 2D
    h2 = h.contiguous().view(-1, h.shape[-1])              # [M, K]
    r2 = residual.contiguous().view(-1, residual.shape[-1])# [M, N]
    M, K = h2.shape
    N = weight.shape[0]

    y2 = torch.empty((M, N), device=h.device, dtype=h.dtype)

    grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))

    in_dtype = tl.float16 if h.dtype == torch.float16 else tl.bfloat16

    fused_linear_bias_residual_kernel[grid](
        h2, weight, bias, r2, y2,
        M=M, N=N, K=K,
        stride_hm=h2.stride(0), stride_hk=h2.stride(1),
        stride_wn=weight.stride(0), stride_wk=weight.stride(1),
        stride_rm=r2.stride(0), stride_rn=r2.stride(1),
        stride_ym=y2.stride(0), stride_yn=y2.stride(1),
        IN_DTYPE=in_dtype,
    )

    return y2.view(*h.shape[:-1], N)
