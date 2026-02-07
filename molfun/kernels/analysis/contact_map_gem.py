"""
molfun/kernels/analysis/contact_map_gem.py

Optimized Triton kernel for computing atomic contact maps with bitpacked output.

=============================================================================
KERNEL ARCHITECTURE (1D Tiled with Vectorized Bitpacking)
=============================================================================

This kernel uses a different strategy than contact_map_atoms.py:

1. 1D GRID (row blocks)
   - Each program handles BLOCK_SIZE_ROW rows
   - Iterates over all column bytes within the program
   - Better for cases where N is moderate and we want to maximize row parallelism

2. VECTORIZED 8-COLUMN PROCESSING
   - Loads 8 columns at a time (one output byte worth)
   - Broadcasting: xi[:, None] - xj[None, :] => [BLOCK_ROW, 8]
   - All 8 bits computed in parallel

3. ARITHMETIC BITPACKING
   - Uses powers of 2: [1, 2, 4, 8, 16, 32, 64, 128]
   - Multiply contacts by powers and sum => packed byte
   - No branches, pure arithmetic

4. MEMORY ACCESS PATTERN
   - Row coordinates loaded once per program
   - Column coordinates loaded per byte (8 values)
   - Output written one byte at a time

=============================================================================
PERFORMANCE CHARACTERISTICS
=============================================================================

- Good for N up to ~10k atoms
- Better memory reuse of row coordinates
- Each program does more work but needs fewer programs
- Tradeoff: less parallelism but better cache utilization

=============================================================================
"""

import torch
import triton
import triton.language as tl


@triton.jit
def contact_map_kernel_tiled(
    x_ptr, y_ptr, z_ptr,
    out_ptr,
    N,
    n_bytes,
    out_stride_row,
    cutoff2,
    BLOCK_SIZE_ROW: tl.constexpr
):
    """
    Optimized kernel with tiling and vectorization.
    
    - Processes a block of rows (BLOCK_SIZE_ROW) at a time
    - Vectorizes 8 columns simultaneously to pack one byte at once
    - Uses arithmetic bitpacking instead of bit shifts
    """
    # Row block identifier (e.g., rows 0-63, 64-127, ...)
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_ROW
    
    # Offsets for rows this program will process
    rows_offs = row_start + tl.arange(0, BLOCK_SIZE_ROW)
    mask_rows = rows_offs < N

    # Load row coordinates [BLOCK_SIZE_ROW]
    xi = tl.load(x_ptr + rows_offs, mask=mask_rows, other=0.0)
    yi = tl.load(y_ptr + rows_offs, mask=mask_rows, other=0.0)
    zi = tl.load(z_ptr + rows_offs, mask=mask_rows, other=0.0)

    # Convert to float32 for precision
    xi = xi.to(tl.float32)
    yi = yi.to(tl.float32)
    zi = zi.to(tl.float32)

    # Powers of 2 for arithmetic bitpacking: [1, 2, 4, 8, 16, 32, 64, 128]
    # Shape: [1, 8] for broadcasting against rows
    powers = tl.arange(0, 8)
    powers = (1 << powers).to(tl.uint8)
    powers = powers[None, :]  # [1, 8]

    # Iterate over columns in steps of 8 (1 output byte per iteration)
    for byte_idx in range(n_bytes):
        # Indices of the 8 current columns
        base_col = byte_idx * 8
        cols_offs = base_col + tl.arange(0, 8)
        mask_cols = cols_offs < N

        # Load block of 8 columns [8]
        xj = tl.load(x_ptr + cols_offs, mask=mask_cols, other=0.0).to(tl.float32)
        yj = tl.load(y_ptr + cols_offs, mask=mask_cols, other=0.0).to(tl.float32)
        zj = tl.load(z_ptr + cols_offs, mask=mask_cols, other=0.0).to(tl.float32)

        # Broadcasting magic:
        # xi: [BLOCK_ROW] -> [BLOCK_ROW, 1]
        # xj: [8]         -> [1, 8]
        # Result: [BLOCK_ROW, 8] -> Matrix of differences
        dx = xi[:, None] - xj[None, :]
        dy = yi[:, None] - yj[None, :]
        dz = zi[:, None] - zj[None, :]

        dist2 = dx * dx + dy * dy + dz * dz

        # Boolean comparison [BLOCK_ROW, 8]
        # Also exclude diagonal (row != col)
        is_contact = (dist2 < cutoff2) & (rows_offs[:, None] != cols_offs[None, :])
        
        # Apply edge masks (if N is not multiple of 8 or BLOCK)
        is_contact = is_contact & mask_rows[:, None] & mask_cols[None, :]

        # VECTORIZED BITPACKING
        # Instead of ifs and shifts, multiply by powers of 2 and sum.
        # is_contact (bool) -> uint8 * powers (1, 2, 4...) -> horizontal sum
        # [BLOCK_ROW, 8] * [1, 8] -> [BLOCK_ROW, 8] -> sum(axis=1) -> [BLOCK_ROW]
        packed_byte = tl.sum(is_contact.to(tl.uint8) * powers, axis=1)

        # Store result
        out_offs = rows_offs * out_stride_row + byte_idx
        tl.store(out_ptr + out_offs, packed_byte, mask=mask_rows)


def contact_map_atoms_bitpack_fast(coords: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Compute atomic contact map with bitpacked output using optimized Triton kernel.
    
    This version uses:
    - 1D grid over row blocks
    - Vectorized 8-column processing
    - Arithmetic bitpacking (no branches)
    
    Args:
        coords: [N, 3] CUDA tensor with atom coordinates
        cutoff: Distance cutoff for contacts
    
    Returns:
        Packed contact map [N, ceil(N/8)] as uint8 tensor.
    """
    assert coords.is_cuda, "coords must be on CUDA"
    assert coords.ndim == 2 and coords.shape[1] == 3, "coords must be [N, 3]"
    
    N = coords.shape[0]
    n_bytes = (N + 7) // 8
    
    # Output packed tensor
    out = torch.zeros((N, n_bytes), device=coords.device, dtype=torch.uint8)
    
    x = coords[:, 0].contiguous()
    y = coords[:, 1].contiguous()
    z = coords[:, 2].contiguous()
    
    cutoff2 = float(cutoff * cutoff)
    
    # Tuning: Block size 128 is usually a good sweet spot
    BLOCK_SIZE_ROW = 128
    
    # 1D Grid: Number of blocks needed to cover N rows
    grid = (triton.cdiv(N, BLOCK_SIZE_ROW),)
    
    contact_map_kernel_tiled[grid](
        x, y, z,
        out,
        N,
        n_bytes,
        out.stride(0),
        cutoff2,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW
    )
    
    return out
