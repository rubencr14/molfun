"""
This file implements a **fused GELU activation** using **Triton**, meant as an educational,
“first real kernel” for a Transformer MLP path (e.g., ESM / protein LMs).

What problem are we solving?
- In a typical Transformer MLP you compute something like:
    h = linear1(x)          # big GEMM (usually includes bias)
    h = GELU(h)             # activation
    y = linear2(h)          # big GEMM
- Framework code often ends up launching separate GPU kernels for the activation step.
  Even when the activation itself is “small” compared to GEMMs, it still causes:
  1) extra kernel launch overhead,
  2) extra memory traffic to/from HBM (global GPU memory) if intermediate tensors are
     written and read back.

What does this Triton kernel do?
- It applies GELU elementwise on a tensor **in one GPU kernel**:
    y = gelu(x)
- We flatten the input tensor to 1D so the kernel can treat it as a contiguous array.
- The kernel is launched as a 1D grid of programs (similar to CUDA blocks):
  each program processes a contiguous chunk of `BLOCK` elements.

How does the kernel map work to GPU threads?
- `pid = tl.program_id(0)` gives the program index along axis 0.
- Each program handles indices:
    offsets = pid*BLOCK + [0..BLOCK-1]
- `mask = offsets < n_elements` prevents out-of-bounds access on the last partial block.

Why the fp32 upcast?
- GELU uses non-linear math (tanh) and intermediate values can be sensitive in fp16/bf16.
- We cast x -> fp32 for the internal computation, then cast back to the original dtype.
  This is a common pattern for stability while still storing/returning fp16/bf16.

Which GELU variant is this?
- This uses the popular tanh approximation:
    GELU(x) ≈ 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
- Many models use this approximation because it is fast and differentiable.

What you learn from this kernel (concepts you care about):
- Memory coalescing: loading/storing contiguous regions (`offsets` are contiguous).
- Blocking / tiling: each program processes a tile of size `BLOCK`.
- Masking: safe tail handling for non-multiple-of-BLOCK sizes.
- Mixed precision: fp16/bf16 I/O with fp32 math inside.
- Kernel launch configuration: `grid = ceil_div(n_elements, BLOCK)` and `num_warps`.

What this kernel is NOT (yet):
- It is not the “ultimate MLP kernel” that fuses GEMM + bias + activation.
  That requires a fused matmul epilogue and careful tiling over M/N/K.
- This is intentionally a stepping stone: easy to validate correctness, easy to see in
  Nsight Systems/Compute, and it prepares you for more advanced fusion.

How you would use it in practice:
- Replace an activation call `torch.nn.functional.gelu(h)` with `gelu_triton(h)`
  inside the MLP of an ESM encoder layer.
- First validate numerical closeness vs PyTorch GELU on representative shapes, then
  profile with Nsight to ensure the kernel is executing efficiently.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel_erf(x_ptr, y_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                     # program index (like CUDA block id)
    block_start = pid * BLOCK                       # first element index for this program
    offsets = block_start + tl.arange(0, BLOCK)     # vector of indices handled by this program
    mask = offsets < n_elements                     # guard for tail

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)  # load input tile
    x_f32 = x.to(tl.float32)                            # upcast for stable math

    inv_sqrt2 = 0.7071067811865476                      # 1/sqrt(2)
    u = x_f32 * inv_sqrt2                                # x / sqrt(2)

    # Triton math namespace is version-dependent; tl.math.erf is commonly available.
    # If your Triton exposes tl.erf directly, you can swap to tl.erf(u).
    erf_u = tl.math.erf(u)

    y_f32 = 0.5 * x_f32 * (1.0 + erf_u)                 # exact GELU

    y = y_f32.to(x.dtype)                               # cast back to fp16/bf16
    tl.store(y_ptr + offsets, y, mask=mask)             # store output tile


def gelu_triton(x: torch.Tensor, block: int = 1024) -> torch.Tensor:
    assert x.is_cuda, "x must be a CUDA tensor"
    x_flat = x.contiguous().view(-1)                    # flatten for 1D kernel
    n = x_flat.numel()                                  # total elements
    y = torch.empty_like(x_flat)                        # allocate output

    grid = (triton.cdiv(n, block),)                     # number of programs
    gelu_kernel_erf[grid](
        x_flat, y,
        n_elements=n,
        BLOCK=block,
        num_warps=4
    )

    return y.view_as(x)                                 # reshape back
