"""
molfun/kernels/analysis/pairwise_distance.py

Triton kernel for computing pairwise Euclidean distances between 3D coordinates.

This kernel computes the N x N distance matrix for N points in 3D space:
    D[i, j] = sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2 + (z[i] - z[j])^2)

The kernel uses a 2D grid of programs, where each program computes a BLOCK x BLOCK
tile of the output matrix. This allows efficient parallelization for large N.

Input:
    coords: torch.Tensor of shape (N, 3) with x, y, z coordinates

Output:
    distances: torch.Tensor of shape (N, N) with pairwise distances
"""

import torch
import triton
import triton.language as tl


@triton.jit
def pairwise_distance_kernel(
    x_ptr, y_ptr, z_ptr,          # coords separados
    out_ptr,                      # matriz N x N
    N: tl.constexpr,
    stride_out: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Compute pairwise Euclidean distances for 3D coordinates.
    
    Each program handles a BLOCK x BLOCK tile of the output matrix.
    """
    pid_i = tl.program_id(0)  # row block index
    pid_j = tl.program_id(1)  # column block index

    # Compute column indices for this tile (loaded once, reused for all rows)
    col_offsets = pid_j * BLOCK + tl.arange(0, BLOCK)
    col_mask = col_offsets < N
    
    # Load coordinates for columns (j) - these will be reused for all rows
    xj = tl.load(x_ptr + col_offsets, mask=col_mask, other=0.0)
    yj = tl.load(y_ptr + col_offsets, mask=col_mask, other=0.0)
    zj = tl.load(z_ptr + col_offsets, mask=col_mask, other=0.0)

    # Compute pairwise distances for each row i in the tile
    # For each row, we compute distances to all columns j using vectorized operations
    for i in range(BLOCK):
        # Calculate row index directly (can't index into tensor in Triton loops)
        row_idx = pid_i * BLOCK + i
        
        # Check if this row index is valid (Triton doesn't support continue)
        if row_idx < N:
            # Load coordinates for row i (scalars)
            xi_val = tl.load(x_ptr + row_idx)
            yi_val = tl.load(y_ptr + row_idx)
            zi_val = tl.load(z_ptr + row_idx)
            
            # Broadcast row coordinates to match column vector length
            # Compute differences: dx[j] = xi[i] - xj[j] for all j
            dx = xi_val - xj  # vectorized: scalar - vector
            dy = yi_val - yj  # vectorized: scalar - vector
            dz = zi_val - zj  # vectorized: scalar - vector
            
            # Compute squared distances (vectorized)
            dist_sq = dx * dx + dy * dy + dz * dz
            
            # Compute distances (vectorized sqrt)
            dist = tl.sqrt(dist_sq)
            
            # Compute output pointers for this row
            out_ptrs = out_ptr + row_idx * stride_out + col_offsets
            
            # Store results for this row (vectorized store)
            tl.store(out_ptrs, dist, mask=col_mask)


def pairwise_distances_triton(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances for 3D coordinates using Triton.
    
    Args:
        coords: Tensor of shape (N, 3) with x, y, z coordinates
        
    Returns:
        distances: Tensor of shape (N, N) with pairwise distances
    """
    assert coords.dim() == 2 and coords.shape[1] == 3, "coords must be (N, 3)"
    assert coords.is_cuda, "coords must be on CUDA"
    
    N = coords.shape[0]
    dtype = coords.dtype
    
    # Split coordinates into separate arrays
    x = coords[:, 0].contiguous()
    y = coords[:, 1].contiguous()
    z = coords[:, 2].contiguous()
    
    # Allocate output
    out = torch.empty((N, N), device=coords.device, dtype=dtype)
    
    # Launch configuration
    BLOCK = 64  # Tile size (adjust based on GPU and N)
    grid = (
        triton.cdiv(N, BLOCK),  # number of row blocks
        triton.cdiv(N, BLOCK),  # number of column blocks
    )
    
    # Launch kernel
    pairwise_distance_kernel[grid](
        x, y, z, out,
        N=N,
        stride_out=out.stride(0),
        BLOCK=BLOCK,
    )
    
    return out
