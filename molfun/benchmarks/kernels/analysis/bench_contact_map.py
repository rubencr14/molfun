"""
molfun/benchmarks/bench_contact_map.py

Benchmark contact map computation: Triton bitpacked vs PyTorch naive.

This benchmark compares:
- Triton: Bitpacked contact map kernel (contact_map_atoms_bitpack)
- PyTorch: Naive pairwise distance + threshold implementation

The kernel computes which atom pairs are within a distance cutoff:
    C[i,j] = 1 if ||r_i - r_j|| < cutoff, else 0

Usage:
    python molfun/benchmarks/bench_contact_map.py

Notes:
- Uses CUDA events for accurate timing with proper synchronization
- Includes correctness checks against PyTorch reference
- Tests various sizes N (number of atoms) and cutoffs
- Triton version uses bitpacking (8x memory reduction)
"""

import torch
import time
from typing import List, Dict, Any

from molfun.kernels.analysis.contact_map_atoms import (
    contact_map_atoms_bitpack,
    contact_map_pytorch,
)


def unpack_bitpack_correct(packed: torch.Tensor, N: int) -> torch.Tensor:
    """
    Correctly unpack bitpacked contact map to dense boolean matrix.
    
    Args:
        packed: [N, ceil(N/8)] uint8 tensor
        N: Number of atoms
    
    Returns:
        [N, N] bool tensor
    """
    device = packed.device
    n_bytes = packed.shape[1]
    
    # Create bit positions [0, 1, 2, ..., 7]
    bits = torch.arange(8, device=device, dtype=torch.uint8)
    
    # Expand packed: [N, n_bytes, 1] >> [1, 1, 8] -> [N, n_bytes, 8]
    expanded = ((packed.unsqueeze(-1) >> bits) & 1).to(torch.bool)
    
    # Reshape to [N, n_bytes * 8] and trim to [N, N]
    dense = expanded.reshape(N, -1)[:, :N]
    
    return dense


def count_bits_packed(packed: torch.Tensor) -> int:
    """Count total number of 1 bits in packed tensor."""
    # Lookup table for popcount
    popcount_lut = torch.tensor(
        [bin(i).count("1") for i in range(256)],
        device=packed.device,
        dtype=torch.int32
    )
    return popcount_lut[packed.to(torch.int64)].sum().item()


def time_it_cuda_sync(fn, iters: int, warmup: int) -> float:
    """
    Time a GPU function with PROPER synchronization.
    
    Critical: We synchronize AFTER each call to ensure the kernel
    actually completes, not just launches.
    """
    # Warmup
    for _ in range(warmup):
        result = fn()
        torch.cuda.synchronize()  # Wait for kernel to complete
    
    # Timing with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start.record()
    
    for _ in range(iters):
        result = fn()
    
    end.record()
    torch.cuda.synchronize()  # Wait for all kernels to complete
    
    ms_total = start.elapsed_time(end)
    return ms_total / iters


def run_benchmark() -> List[Dict[str, Any]]:
    """Execute benchmark and return structured results."""
    assert torch.cuda.is_available(), "CUDA required"
    torch.set_grad_enabled(False)

    # Benchmark cases: (N atoms, cutoff in Angstroms)
    cases = [
        (500, 8.0),
        (1000, 8.0),
        (2000, 8.0),
        (5000, 8.0),
        (10000, 8.0),
        (1000, 4.0),   # Sparse contacts
        (1000, 12.0),  # Dense contacts
    ]

    results = []
    
    print("Contact Map Benchmark: Triton Bitpack vs PyTorch")
    print("=" * 110)

    for N, cutoff in cases:
        # Generate random 3D coordinates (simulating protein structure)
        coords = torch.randn(N, 3, device="cuda", dtype=torch.float32) * 10

        # --- PyTorch (naive) timing ---
        def pytorch_fn():
            return contact_map_pytorch(coords, cutoff)
        
        t_pytorch = time_it_cuda_sync(pytorch_fn, iters=20, warmup=5)

        # --- Triton bitpack timing ---
        # Pre-trigger JIT compilation
        _ = contact_map_atoms_bitpack(coords, cutoff)
        torch.cuda.synchronize()

        def triton_fn():
            return contact_map_atoms_bitpack(coords, cutoff)
        
        t_triton = time_it_cuda_sync(triton_fn, iters=20, warmup=5)

        # --- Correctness check ---
        # Get outputs
        cm_pytorch = contact_map_pytorch(coords, cutoff)  # [N, N] bool, full matrix
        cm_triton_packed = contact_map_atoms_bitpack(coords, cutoff)  # [N, ceil(N/8)] uint8
        
        # Unpack Triton output to dense
        cm_triton_dense = unpack_bitpack_correct(cm_triton_packed, N)
        
        # Triton only computes upper triangle, so compare only upper triangle
        upper_mask = torch.triu(torch.ones(N, N, device="cuda", dtype=torch.bool), diagonal=1)
        
        # Mask both to upper triangle for comparison
        pytorch_upper = cm_pytorch & upper_mask
        triton_upper = cm_triton_dense & upper_mask
        
        # Count mismatches
        mismatches = (pytorch_upper != triton_upper).sum().item()
        total_upper_pairs = (N * (N - 1)) // 2
        
        # Count contacts
        pytorch_contacts = pytorch_upper.sum().item()
        triton_contacts = triton_upper.sum().item()
        
        # Alternative: count bits directly from packed
        triton_bits_total = count_bits_packed(cm_triton_packed)
        
        # Sparsity
        sparsity = pytorch_contacts / total_upper_pairs if total_upper_pairs > 0 else 0

        # Memory comparison
        mem_pytorch = N * N  # bytes (bool)
        mem_triton = N * ((N + 7) // 8)  # bytes (bitpacked)
        mem_ratio = mem_pytorch / mem_triton

        speedup = t_pytorch / t_triton if t_triton > 0 else float("inf")

        result = {
            "benchmark_name": "contact_map",
            "case_name": f"N={N}_cutoff={cutoff}",
            "baseline_time_ms": round(t_pytorch, 4),
            "triton_time_ms": round(t_triton, 4),
            "speedup": round(speedup, 2) if speedup != float("inf") else None,
            "max_diff": mismatches,
            "mean_diff": mismatches / total_upper_pairs if total_upper_pairs > 0 else 0,
            "metadata": {
                "N": N,
                "cutoff": cutoff,
                "pytorch_contacts": pytorch_contacts,
                "triton_contacts": triton_contacts,
                "triton_bits_total": triton_bits_total,
                "mismatches": mismatches,
                "sparsity_pct": round(sparsity * 100, 2),
                "mem_pytorch_mb": round(mem_pytorch / 1e6, 3),
                "mem_triton_mb": round(mem_triton / 1e6, 3),
                "mem_ratio": round(mem_ratio, 1),
                "dtype": str(coords.dtype),
            }
        }
        results.append(result)

        # Print summary
        match_str = "✓" if mismatches == 0 else f"✗ {mismatches}"
        print(f"N={N:5d} cut={cutoff:.1f}Å | PyTorch: {t_pytorch:8.3f}ms | "
              f"Triton: {t_triton:8.3f}ms | Speedup: {speedup:6.2f}x | "
              f"Contacts: {pytorch_contacts:>8,} vs {triton_contacts:>8,} | "
              f"Match: {match_str}")

    return results


def main():
    results = run_benchmark()
    
    print("\n" + "=" * 110)
    print("Contact Map Benchmark Summary")
    print("=" * 110)
    print(f"{'Case':<22} | {'PyTorch':>10} | {'Triton':>10} | {'Speedup':>8} | "
          f"{'Contacts':>10} | {'Mismatches':>10} | {'Mem':>6}")
    print("-" * 110)
    
    for result in results:
        meta = result["metadata"]
        case = f"N={meta['N']}, cut={meta['cutoff']}Å"
        print(f"{case:<22} | {result['baseline_time_ms']:10.3f} | {result['triton_time_ms']:10.3f} | "
              f"{result['speedup']:7.2f}x | {meta['pytorch_contacts']:>10,} | "
              f"{meta['mismatches']:>10} | {meta['mem_ratio']:5.0f}x")


if __name__ == "__main__":
    main()
