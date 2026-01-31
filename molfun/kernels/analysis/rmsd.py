"""
molfun/kernels/analysis/rmsd.py

Triton kernels for computing RMSD (Root Mean Square Deviation).

RMSD is defined as:
    RMSD = sqrt( (1/N) * sum_i( |r_A[i] - r_B[i]|^2 ) )

This module provides:
- rmsd_triton: Single RMSD computation (returns GPU tensor)
- rmsd_batch_triton: Batch RMSD computation (optimized kernel)
"""

import torch
import triton
import triton.language as tl


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
    Each program handles BLOCK atoms and stores the sum to partial results.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load coordinates
    xA = tl.load(xA_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    yA = tl.load(yA_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    zA = tl.load(zA_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    xB = tl.load(xB_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    yB = tl.load(yB_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    zB = tl.load(zB_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute squared differences
    dx = xA - xB
    dy = yA - yB
    dz = zA - zB
    dist2 = dx * dx + dy * dy + dz * dz

    # Sum within this block
    acc = tl.sum(dist2, axis=0)
    tl.store(partial_ptr + pid, acc)


@triton.jit
def rmsd_batch_kernel_1d(
    ref_ptr,       # [N, 3] reference coordinates
    batch_ptr,     # [M, N, 3] batch coordinates (MUST be contiguous)
    out_ptr,       # [M] output RMSD values
    N,
    stride_ref0: tl.constexpr,   # ref stride for atoms (N dimension)
    stride_ref1: tl.constexpr,   # ref stride for xyz (3 dimension)
    stride_b0: tl.constexpr,     # batch stride for frames (M dimension)
    stride_b1: tl.constexpr,     # batch stride for atoms (N dimension)
    stride_b2: tl.constexpr,     # batch stride for xyz (3 dimension)
    BLOCK: tl.constexpr,
):
    """
    Batch RMSD kernel: 1 program per frame (1D grid).
    
    Each program:
    - Processes one complete frame
    - Loops over N atoms in blocks
    - Accumulates in FP32
    - Writes a single RMSD value
    
    This design avoids the offset/stride bugs of the 2D grid approach.
    """
    m = tl.program_id(0)  # frame index
    
    # Accumulator in FP32 for precision
    acc = tl.zeros((), dtype=tl.float32)
    
    # Process N atoms in chunks of BLOCK
    for start in range(0, N, BLOCK):
        i = start + tl.arange(0, BLOCK)
        mask = i < N
        
        # Load reference coordinates [N, 3]
        # Address: ref_ptr + i * stride_ref0 + coord_idx * stride_ref1
        rx = tl.load(ref_ptr + i * stride_ref0 + 0 * stride_ref1, mask=mask, other=0.0).to(tl.float32)
        ry = tl.load(ref_ptr + i * stride_ref0 + 1 * stride_ref1, mask=mask, other=0.0).to(tl.float32)
        rz = tl.load(ref_ptr + i * stride_ref0 + 2 * stride_ref1, mask=mask, other=0.0).to(tl.float32)
        
        # Load batch coordinates [M, N, 3]
        # Address: batch_ptr + m * stride_b0 + i * stride_b1 + coord_idx * stride_b2
        bx = tl.load(batch_ptr + m * stride_b0 + i * stride_b1 + 0 * stride_b2, mask=mask, other=0.0).to(tl.float32)
        by = tl.load(batch_ptr + m * stride_b0 + i * stride_b1 + 1 * stride_b2, mask=mask, other=0.0).to(tl.float32)
        bz = tl.load(batch_ptr + m * stride_b0 + i * stride_b1 + 2 * stride_b2, mask=mask, other=0.0).to(tl.float32)
        
        # Squared differences
        dx = rx - bx
        dy = ry - by
        dz = rz - bz
        
        # Accumulate sum of squared distances
        acc += tl.sum(dx * dx + dy * dy + dz * dz, axis=0)
    
    # Compute RMSD and store
    rmsd = tl.sqrt(acc / tl.full((), N, tl.float32))
    tl.store(out_ptr + m, rmsd)


def rmsd_triton(coords_A: torch.Tensor, coords_B: torch.Tensor) -> torch.Tensor:
    """
    Compute RMSD between two sets of 3D coordinates using Triton.
    
    Returns GPU tensor (no CPU sync).
    
    Args:
        coords_A: Tensor of shape (N, 3) - first structure coordinates
        coords_B: Tensor of shape (N, 3) - second structure coordinates
        
    Returns:
        rmsd: Tensor of shape () on GPU - RMSD value
    """
    assert coords_A.shape == coords_B.shape, "Coordinate shapes must match"
    assert coords_A.dim() == 2 and coords_A.shape[1] == 3, "coords must be (N, 3)"
    assert coords_A.is_cuda and coords_B.is_cuda, "coords must be on CUDA"
    
    N = coords_A.shape[0]
    
    # Split coordinates
    xA = coords_A[:, 0].contiguous()
    yA = coords_A[:, 1].contiguous()
    zA = coords_A[:, 2].contiguous()
    
    xB = coords_B[:, 0].contiguous()
    yB = coords_B[:, 1].contiguous()
    zB = coords_B[:, 2].contiguous()
    
    # Launch configuration
    BLOCK = 1024
    num_blocks = triton.cdiv(N, BLOCK)
    
    # Allocate partial sums (float32 for precision)
    partial_sums = torch.zeros(num_blocks, device=coords_A.device, dtype=torch.float32)
    
    # Launch kernel
    rmsd_partial_kernel[(num_blocks,)](
        xA, yA, zA,
        xB, yB, zB,
        partial_sums,
        N,
        BLOCK=BLOCK,
    )
    
    # Compute final RMSD on GPU (no CPU sync)
    total_sum = partial_sums.sum()  # GPU tensor
    rmsd = torch.sqrt(total_sum / N)  # GPU tensor
    
    return rmsd


def rmsd_batch_triton(coords_A: torch.Tensor, coords_B_batch: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """
    Compute RMSD between a reference structure and a batch of structures.
    
    Uses 1D grid (1 program per frame) for robust memory access.
    
    Args:
        coords_A: Tensor of shape (N, 3) - reference structure coordinates
        coords_B_batch: Tensor of shape (M, N, 3) - batch of M structures
        block: Block size for processing atoms (default 1024)
        
    Returns:
        rmsds: Tensor of shape (M,) on GPU - RMSD values for each structure
    """
    assert coords_A.is_cuda and coords_B_batch.is_cuda, "coords must be on CUDA"
    assert coords_A.dim() == 2 and coords_A.shape[1] == 3, "coords_A must be (N, 3)"
    assert coords_B_batch.dim() == 3 and coords_B_batch.shape[2] == 3, "coords_B_batch must be (M, N, 3)"
    assert coords_A.shape[0] == coords_B_batch.shape[1], "Number of atoms must match"
    
    # CRITICAL: ensure contiguous to avoid stride bugs
    ref_c = coords_A.contiguous()
    batch_c = coords_B_batch.contiguous()
    
    M, N, _ = batch_c.shape
    
    # Output: one RMSD per frame
    out = torch.empty((M,), device=coords_A.device, dtype=torch.float32)
    
    # 1D grid: one program per frame
    grid = (M,)
    
    # Launch kernel with proper strides
    rmsd_batch_kernel_1d[grid](
        ref_c,
        batch_c,
        out,
        N,
        stride_ref0=ref_c.stride(0),
        stride_ref1=ref_c.stride(1),
        stride_b0=batch_c.stride(0),
        stride_b1=batch_c.stride(1),
        stride_b2=batch_c.stride(2),
        BLOCK=block,
        num_warps=4,
    )
    
    return out
