"""
molfun/kernels/analysis/rmsd.py

Triton kernel for computing RMSD (Root Mean Square Deviation) between two sets of 3D coordinates.

RMSD is defined as:
    RMSD = sqrt( (1/N) * sum_i( |r_A[i] - r_B[i]|^2 ) )

Where r_A and r_B are the 3D coordinates of the two structures.

The kernel uses a two-phase approach:
1. Compute partial sums of squared distances in parallel (one per block)
2. Sum the partial results on CPU and compute final RMSD

Input:
    coords_A: torch.Tensor of shape (N, 3) - first structure coordinates
    coords_B: torch.Tensor of shape (N, 3) - second structure coordinates

Output:
    rmsd: float - RMSD value
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def rmsd_partial_kernel(
    xA_ptr, yA_ptr, zA_ptr,
    xB_ptr, yB_ptr, zB_ptr,
    partial_ptr,
    N,
    BLOCK: tl.constexpr,
):
    """
    Compute partial sums of squared distances for RMSD calculation.
    
    Each program handles BLOCK atoms and stores the sum of squared distances
    to a partial results array. The final RMSD is computed on CPU.
    """
    pid = tl.program_id(0)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load coordinates for structure A
    xA = tl.load(xA_ptr + offs, mask=mask, other=0.0)
    yA = tl.load(yA_ptr + offs, mask=mask, other=0.0)
    zA = tl.load(zA_ptr + offs, mask=mask, other=0.0)

    # Load coordinates for structure B
    xB = tl.load(xB_ptr + offs, mask=mask, other=0.0)
    yB = tl.load(yB_ptr + offs, mask=mask, other=0.0)
    zB = tl.load(zB_ptr + offs, mask=mask, other=0.0)

    # Compute squared differences
    dx = xA - xB
    dy = yA - yB
    dz = zA - zB

    dist2 = dx * dx + dy * dy + dz * dz

    # Sum within this block
    acc = tl.sum(dist2, axis=0)

    # Store partial sum
    tl.store(partial_ptr + pid, acc)


def rmsd_triton(coords_A: torch.Tensor, coords_B: torch.Tensor) -> float:
    """
    Compute RMSD between two sets of 3D coordinates using Triton.
    
    Args:
        coords_A: Tensor of shape (N, 3) - first structure coordinates
        coords_B: Tensor of shape (N, 3) - second structure coordinates
        
    Returns:
        rmsd: float - RMSD value
    """
    assert coords_A.shape == coords_B.shape, "Coordinate shapes must match"
    assert coords_A.dim() == 2 and coords_A.shape[1] == 3, "coords must be (N, 3)"
    assert coords_A.is_cuda and coords_B.is_cuda, "coords must be on CUDA"
    
    N = coords_A.shape[0]
    
    # Split coordinates into separate arrays
    xA = coords_A[:, 0].contiguous()
    yA = coords_A[:, 1].contiguous()
    zA = coords_A[:, 2].contiguous()
    
    xB = coords_B[:, 0].contiguous()
    yB = coords_B[:, 1].contiguous()
    zB = coords_B[:, 2].contiguous()
    
    # Launch configuration
    BLOCK = 1024
    num_blocks = triton.cdiv(N, BLOCK)
    
    # Allocate partial sums array
    partial_sums = torch.zeros(num_blocks, device=coords_A.device, dtype=coords_A.dtype)
    
    # Launch kernel
    rmsd_partial_kernel[(num_blocks,)](
        xA, yA, zA,
        xB, yB, zB,
        partial_sums,
        N,
        BLOCK=BLOCK,
    )
    
    # Compute final RMSD on CPU
    total_sum = partial_sums.sum().item()
    rmsd = math.sqrt(total_sum / N)
    
    return rmsd


def rmsd_batch_triton(coords_A: torch.Tensor, coords_B_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute RMSD between a reference structure and a batch of structures.
    
    Args:
        coords_A: Tensor of shape (N, 3) - reference structure coordinates
        coords_B_batch: Tensor of shape (M, N, 3) - batch of M structures
        
    Returns:
        rmsds: Tensor of shape (M,) - RMSD values for each structure in batch
    """
    assert coords_A.dim() == 2 and coords_A.shape[1] == 3, "coords_A must be (N, 3)"
    assert coords_B_batch.dim() == 3 and coords_B_batch.shape[2] == 3, "coords_B_batch must be (M, N, 3)"
    assert coords_A.shape[0] == coords_B_batch.shape[1], "Number of atoms must match"
    
    M = coords_B_batch.shape[0]
    rmsds = torch.zeros(M, device=coords_A.device, dtype=coords_A.dtype)
    
    for i in range(M):
        rmsds[i] = rmsd_triton(coords_A, coords_B_batch[i])
    
    return rmsds
