"""
Fused Linear + Bias + GELU (erf-based) using Triton.

Goal
-----
Compute in ONE kernel:
    Y = GELU( X @ W^T + b )

Where:
- X shape: [M, K]   (we flatten batch/sequence into M)
- W shape: [N, K]   (PyTorch nn.Linear stores weight as [out_features, in_features])
- b shape: [N]
- Y shape: [M, N,]

Why this matters for ESM MLP
----------------------------
In an ESM encoder MLP block you typically have:
  h = dense1(x)         # shape: [B, T, 4D]
  h = GELU(h)
  y = dense2(h)         # shape: [B, T, D]

Even if cuBLAS is very fast for GEMM, the overall MLP often pays extra for:
- writing the intermediate output of dense1 to global memory (HBM)
- reading it back for GELU
- writing again after GELU
- kernel launch overhead for GELU

By fusing GEMM + bias + GELU:
- we reduce global memory traffic (fewer round trips to HBM)
- we reduce kernel launches
- we can often improve end-to-end latency for "medium" GEMM sizes (ESM-small is exactly that)

Numerical notes
---------------
- We accumulate in fp32 for stability.
- We apply GELU using the "exact" erf form:
    GELU(x) = 0.5*x*(1 + erf(x/sqrt(2)))
  This matches torch.nn.functional.gelu(x, approximate="none")

Compatibility note
------------------
Some Triton versions lack tl.tanh. This implementation uses tl.math.erf.
If your Triton version does not expose tl.math.erf, switch it to tl.erf.
"""

import torch                                   # PyTorch tensors and allocations
import triton                                  # Triton runtime/JIT
import triton.language as tl                   # Triton language primitives


# Autotune: try a few tilings and let Triton pick the fastest per (M,N,K) at runtime.
# These configs are intentionally small and safe for a first iteration.
@triton.autotune(
    configs=[
        triton.Config({"BM": 64,  "BN": 64,  "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 128, "BN": 64,  "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64,  "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64,  "BN": 64,  "BK": 64}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],                       # autotune key based on problem sizes
)
@triton.jit
def fused_linear_bias_gelu_kernel(
    x_ptr,                                     # pointer to X data
    w_ptr,                                     # pointer to W data (shape [N, K])
    b_ptr,                                     # pointer to bias b (shape [N])
    y_ptr,                                     # pointer to output Y
    M: tl.constexpr,                           # number of rows in X (flattened tokens)
    N: tl.constexpr,                           # output features
    K: tl.constexpr,                           # input features
    stride_xm: tl.constexpr,                   # stride for X along M dimension
    stride_xk: tl.constexpr,                   # stride for X along K dimension
    stride_wn: tl.constexpr,                   # stride for W along N dimension (rows)
    stride_wk: tl.constexpr,                   # stride for W along K dimension (cols)
    stride_ym: tl.constexpr,                   # stride for Y along M dimension
    stride_yn: tl.constexpr,                   # stride for Y along N dimension
    BM: tl.constexpr,                          # block size along M
    BN: tl.constexpr,                          # block size along N
    BK: tl.constexpr,                          # block size along K
):
    # --- Program IDs (2D grid) -------------------------------------------------
    # pid_m selects which block of rows (M) we compute
    pid_m = tl.program_id(axis=0)
    # pid_n selects which block of columns (N) we compute
    pid_n = tl.program_id(axis=1)

    # --- Row/col indices this program is responsible for ----------------------
    # row indices: [pid_m*BM .. pid_m*BM + BM-1]
    offs_m = pid_m * BM + tl.arange(0, BM)
    # col indices: [pid_n*BN .. pid_n*BN + BN-1]
    offs_n = pid_n * BN + tl.arange(0, BN)

    # --- Create accumulator in fp32 ------------------------------------------
    # acc is [BM, BN] containing partial sums of dot products
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # --- Loop over K dimension in tiles of BK --------------------------------
    # We iterate k from 0..K in steps of BK and accumulate X_tile @ W_tile^T
    for k0 in range(0, K, BK):
        # k indices for this tile: [k0 .. k0+BK-1]
        offs_k = k0 + tl.arange(0, BK)

        # mask for X loads: valid if row < M and k < K
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

        # Load X tile of shape [BM, BK]
        # X pointer arithmetic: x_ptr + m*stride_xm + k*stride_xk
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=x_mask,
            other=0.0,
        ).to(tl.float16)  # keep as fp16/bf16 in registers; accumulation is fp32

        # mask for W loads: valid if n < N and k < K
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        # Load W tile as [BN, BK] from W[n, k]
        w_nbk = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=w_mask,
            other=0.0,
        ).to(tl.float16)

        # We need W^T shape [BK, BN] for tl.dot with x [BM,BK]
        # So we transpose the loaded tile [BN,BK] -> [BK,BN]
        w = tl.trans(w_nbk)

        # Matrix multiply accumulate: acc += x @ w
        # tl.dot does a dot-product on the BK dimension for each (m,n)
        acc += tl.dot(x, w)

    # --- Bias add --------------------------------------------------------------
    # Load bias for columns offs_n. bias shape is [N]
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

    # Broadcast bias across BM rows and add to acc
    z = acc + b[None, :]

    # --- GELU (exact, erf-based) ------------------------------------------------
    # GELU(z) = 0.5*z*(1 + erf(z/sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    u = z * inv_sqrt2

    # Triton erf function name can vary by version.
    # Most commonly: tl.math.erf. If it fails, switch to tl.erf.
    erf_u = tl.math.erf(u)

    y_f32 = 0.5 * z * (1.0 + erf_u)

    # Cast output back to fp16/bf16 (match X dtype typically)
    y = y_f32.to(tl.float16)

    # --- Store output tile ------------------------------------------------------
    # mask for Y stores: valid if row < M and col < N
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Store Y[m,n]
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        y,
        mask=y_mask,
    )


def fused_linear_gelu_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper that:
    - reshapes x to 2D [M,K]
    - launches the Triton kernel
    - reshapes output back to original batch/sequence shape

    Requirements:
    - inference only (no backward)
    - x: [*, K]
    - weight: [N, K] (nn.Linear)
    - bias: [N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    assert x.dtype in (torch.float16, torch.bfloat16), "Use fp16/bf16 for this kernel"
    assert weight.dtype == x.dtype and bias.dtype == x.dtype, "Match dtypes"
    assert weight.ndim == 2 and bias.ndim == 1
    assert weight.shape[0] == bias.shape[0], "weight out_features must match bias"
    assert x.shape[-1] == weight.shape[1], "x in_features must match weight in_features"

    # Flatten x to [M, K] where M is number of tokens (B*T) and K is hidden dim
    x2 = x.contiguous().view(-1, x.shape[-1])
    M, K = x2.shape
    N = weight.shape[0]

    # Allocate output [M, N]
    y2 = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Grid is 2D: one dimension for M blocks and one for N blocks
    # We pass a lambda so Triton can access the autotuned BM/BN values
    grid = lambda META: (triton.cdiv(M, META["BM"]), triton.cdiv(N, META["BN"]))

    fused_linear_bias_gelu_kernel[grid](
        x2, weight, bias, y2,
        M=M, N=N, K=K,
        stride_xm=x2.stride(0), stride_xk=x2.stride(1),
        stride_wn=weight.stride(0), stride_wk=weight.stride(1),
        stride_ym=y2.stride(0), stride_yn=y2.stride(1),
    )

    # Reshape back: original leading dims + N
    out = y2.view(*x.shape[:-1], N)
    return out
