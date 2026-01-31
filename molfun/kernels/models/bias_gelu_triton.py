# molfun/kernels/bias_gelu_triton.py

# We enable postponed evaluation of annotations (nice-to-have, not required)
from __future__ import annotations

# Torch is used for tensor allocations and calling our kernel
import torch

# Triton is the JIT compiler/runtime for our custom GPU kernel
import triton

# Triton language provides GPU-programming primitives (tl.load, tl.store, tl.arange, etc.)
import triton.language as tl


# This is the actual Triton kernel. Each "program" (think: a CUDA block) will process a tile of elements.
@triton.jit
def bias_gelu_kernel(
    x_ptr,                              # pointer to input tensor x (flattened)
    b_ptr,                              # pointer to bias vector b (size = D)
    y_ptr,                              # pointer to output tensor y (same shape as x)
    n_elements: tl.constexpr,           # total number of elements in x (flattened)
    D: tl.constexpr,                    # last dimension size (hidden size)
    BLOCK: tl.constexpr,                # how many elements this program processes
):
    # Program id: unique integer identifying which "block" we are (like blockIdx.x in CUDA)
    pid = tl.program_id(axis=0)

    # Compute the linear index of the first element handled by this program
    block_start = pid * BLOCK

    # Create a vector of offsets [0, 1, 2, ..., BLOCK-1]
    offsets = block_start + tl.arange(0, BLOCK)

    # Create a mask so we don't read/write out of bounds for the last partial block
    mask = offsets < n_elements

    # Load a block of input elements from global memory (HBM)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute each element's column index within the last dimension (0..D-1)
    # Because we flattened [B, T, D] into 1D, the column is (offset % D)
    col = offsets % D

    # Load the bias value for each element based on its column
    b = tl.load(b_ptr + col, mask=mask, other=0.0)

    # Add bias (this is the first thing we fuse)
    z = x + b

    # --- GELU approximation (tanh-based), widely used and fast ---
    # We upcast to fp32 for stability in intermediate math
    z_f32 = z.to(tl.float32)

    # Compute constant sqrt(2/pi) in fp32
    c = 0.7978845608028654  # sqrt(2/pi)

    # Compute z^3 term used by the tanh approximation
    z3 = z_f32 * z_f32 * z_f32

    # Compute inner term: c * (z + 0.044715*z^3)
    inner = c * (z_f32 + 0.044715 * z3)

    # Compute tanh(inner)
    t = tl.tanh(inner)

    # GELU: 0.5 * z * (1 + tanh(inner))
    y_f32 = 0.5 * z_f32 * (1.0 + t)

    # Cast back to the original dtype of z (typically fp16/bf16)
    y = y_f32.to(z.dtype)

    # Store output back to global memory
    tl.store(y_ptr + offsets, y, mask=mask)


# Python wrapper: checks shapes/dtypes, flattens, launches the kernel, reshapes back.
def bias_gelu_triton(x: torch.Tensor, bias: torch.Tensor, block: int = 1024) -> torch.Tensor:
    # Ensure we are on CUDA because Triton kernels run on GPU
    assert x.is_cuda, "x must be a CUDA tensor"

    # Bias must be on CUDA too
    assert bias.is_cuda, "bias must be a CUDA tensor"

    # Bias must be 1D of length D
    assert bias.ndim == 1, "bias must be a 1D tensor of shape [D]"

    # x must have at least 1 dimension, and last dim must match bias size
    assert x.ndim >= 1, "x must have at least 1 dimension"
    assert x.shape[-1] == bias.shape[0], "x.shape[-1] must match bias.shape[0]"

    # We'll treat x as a flat 1D array for this kernel
    x_flat = x.contiguous().view(-1)

    # D is the hidden size (last dimension)
    D = x.shape[-1]

    # Total number of elements in x
    n_elements = x_flat.numel()

    # Allocate output tensor with same shape and dtype as x
    y = torch.empty_like(x_flat)

    # Compute number of programs (blocks) we need
    grid = (triton.cdiv(n_elements, block),)

    # Launch kernel
    bias_gelu_kernel[grid](
        x_flat,                         # input pointer
        bias,                           # bias pointer
        y,                              # output pointer
        n_elements=n_elements,          # total number of elements
        D=D,                            # hidden size for bias indexing
        BLOCK=block,                    # tile size
        num_warps=4,                    # a reasonable default; tune later
    )

    # Reshape output back to original x shape
    return y.view_as(x)
