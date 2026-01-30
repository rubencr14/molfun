"""
molfun/kernels/analysis/contact_map_atoms.py

Triton kernel for computing atomic contact maps with bitpacked output.

=============================================================================
KERNEL ARCHITECTURE (2D Tiled with Bitpacking)
=============================================================================

The optimized kernel uses:

1. 2D TILING (BM rows × BN cols per program)
   - Each program computes a BM×BN tile of distances
   - Maximizes parallelism across the GPU

2. VECTORIZED LOADS
   - Load BM row coordinates at once (coalesced)
   - Load BN col coordinates at once (coalesced)

3. BROADCASTING FOR DISTANCE COMPUTATION
   - xi[:, None] - xj[None, :] => [BM, BN] differences
   - Compute full BM×BN distance matrix in parallel

4. BITPACKING
   - BN = 8 (one byte per tile column-wise)
   - Output: [N, ceil(N/8)] uint8

=============================================================================
"""

import torch
import triton
import triton.language as tl


@triton.jit
def contact_map_tiled_kernel(
    x_ptr, y_ptr, z_ptr,
    out_ptr,
    N,
    out_stride_row,
    cutoff2,
    BM: tl.constexpr,
):
    """
    2D tiled contact map kernel.
    
    Each program computes BM rows × 8 cols (one byte per row).
    The 8 columns are packed into a single byte.
    """
    pid_row = tl.program_id(0)  # row tile index
    pid_byte = tl.program_id(1)  # byte column index
    
    # Row indices for this tile
    row_start = pid_row * BM
    row_offs = row_start + tl.arange(0, BM)  # [BM]
    row_mask = row_offs < N
    
    # Column indices: 8 columns for this byte
    col_start = pid_byte * 8
    col_offs = col_start + tl.arange(0, 8)  # [8]
    col_mask = col_offs < N
    
    # Load row coordinates (BM values)
    xi = tl.load(x_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)
    yi = tl.load(y_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)
    zi = tl.load(z_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)
    
    # Load col coordinates (8 values)
    xj = tl.load(x_ptr + col_offs, mask=col_mask, other=0.0).to(tl.float32)
    yj = tl.load(y_ptr + col_offs, mask=col_mask, other=0.0).to(tl.float32)
    zj = tl.load(z_ptr + col_offs, mask=col_mask, other=0.0).to(tl.float32)
    
    # Compute pairwise distances: [BM, 8]
    dx = xi[:, None] - xj[None, :]
    dy = yi[:, None] - yj[None, :]
    dz = zi[:, None] - zj[None, :]
    
    dist2 = dx * dx + dy * dy + dz * dz  # [BM, 8]
    
    # Contact matrix
    contacts = dist2 < cutoff2  # [BM, 8]
    
    # Mask diagonal and invalid
    diag_mask = row_offs[:, None] == col_offs[None, :]
    valid_mask = row_mask[:, None] & col_mask[None, :]
    contacts = contacts & ~diag_mask & valid_mask
    
    # Bitpack: pack 8 bits into bytes
    # Each row gets one byte
    contacts_u8 = contacts.to(tl.uint8)  # [BM, 8]
    
    # Create bit weights: [1, 2, 4, 8, 16, 32, 64, 128]
    bit_weights = tl.arange(0, 8).to(tl.uint8)  # [8]
    bit_weights = (1 << bit_weights).to(tl.uint8)  # [8]: [1, 2, 4, ..., 128]
    
    # Weighted sum along columns: contacts_u8 * bit_weights summed
    # [BM, 8] * [8] -> [BM, 8] -> sum -> [BM]
    packed = tl.sum(contacts_u8 * bit_weights[None, :], axis=1).to(tl.uint8)  # [BM]
    
    # Store packed bytes
    out_offs = row_offs * out_stride_row + pid_byte
    tl.store(out_ptr + out_offs, packed, mask=row_mask)


def contact_map_atoms_bitpack(coords: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Compute atomic contact map with bitpacked output using Triton.
    
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

    out = torch.zeros((N, n_bytes), device=coords.device, dtype=torch.uint8)

    x = coords[:, 0].contiguous()
    y = coords[:, 1].contiguous()
    z = coords[:, 2].contiguous()

    cutoff2 = float(cutoff * cutoff)

    # Grid: (row_tiles, byte_cols)
    BM = 64  # rows per tile
    grid = (triton.cdiv(N, BM), n_bytes)
    
    contact_map_tiled_kernel[grid](
        x, y, z,
        out,
        N,
        out.stride(0),
        cutoff2,
        BM=BM,
    )

    return out


def contact_query_bitpack(out_packed: torch.Tensor, i: int, j: int) -> int:
    """Query a single contact from the bitpacked contact map."""
    byte_idx = j >> 3
    bit_idx = j & 7
    v = out_packed[i, byte_idx]
    return int((int(v.item()) >> bit_idx) & 1)


def unpack_contact_map(out_packed: torch.Tensor, N: int) -> torch.Tensor:
    """Unpack bitpacked contact map to full boolean matrix."""
    device = out_packed.device
    bits = torch.arange(8, device=device, dtype=torch.uint8)
    expanded = ((out_packed.unsqueeze(-1) >> bits) & 1).bool()
    dense = expanded.reshape(N, -1)[:, :N]
    return dense


def contact_map_pytorch(coords: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Reference PyTorch implementation using cdist."""
    dist = torch.cdist(coords, coords)
    contacts = dist < cutoff
    contacts.fill_diagonal_(False)
    return contacts
