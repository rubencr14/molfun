"""
molfun/benchmarks/bench_pairwise_distance.py

Benchmark pairwise distance computation: Triton vs PyTorch cdist.

This benchmark compares:
- Baseline: torch.cdist(coords, coords) (PyTorch implementation)
- Triton: pairwise_distances_triton(coords) (custom Triton kernel)

The kernel computes pairwise Euclidean distances for 3D coordinates:
    D[i, j] = sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2 + (z[i] - z[j])^2)

Usage:
    python molfun/benchmarks/bench_pairwise_distance.py

Notes:
- Uses CUDA events for accurate timing
- Includes correctness checks (max/mean absolute difference)
- Tests various sizes N (number of points)
"""

import torch
from typing import List, Dict, Any

from molfun.kernels.analysis.pairwise_distance import pairwise_distances_triton


def time_it_cuda(fn, iters: int, warmup: int) -> float:
    """Time a function using CUDA events"""
    # Warmup (important: triggers Triton JIT/autotune)
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    # CUDA events give accurate GPU timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    torch.cuda.synchronize()

    ms_total = start.elapsed_time(end)
    return ms_total / iters  # ms/iter


def run_benchmark() -> List[Dict[str, Any]]:
    """Ejecuta el benchmark y devuelve resultados estructurados"""
    assert torch.cuda.is_available(), "CUDA required"
    torch.set_grad_enabled(False)

    # Benchmark cases: different numbers of points
    # N is the number of 3D points
    cases = [
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
    ]

    results = []

    for N in cases:
        # Generate random 3D coordinates
        coords = torch.randn(N, 3, device="cuda", dtype=torch.float32)

        # --- Baseline timing (PyTorch cdist) ---
        t_torch = time_it_cuda(
            lambda: torch.cdist(coords, coords),
            iters=50,
            warmup=10
        )

        # --- Triton timing ---
        # Trigger JIT compilation first
        _ = pairwise_distances_triton(coords)
        torch.cuda.synchronize()

        t_triton = time_it_cuda(
            lambda: pairwise_distances_triton(coords),
            iters=50,
            warmup=10
        )

        # --- Correctness check ---
        D_triton = pairwise_distances_triton(coords)
        D_torch = torch.cdist(coords, coords)

        max_diff = (D_triton - D_torch).abs().max().item()
        mean_diff = (D_triton - D_torch).abs().mean().item()

        speedup = t_torch / t_triton if t_triton > 0 else float("inf")

        results.append({
            "benchmark_name": "pairwise_distance",
            "case_name": f"N={N}",
            "baseline_time_ms": round(t_torch, 3),
            "triton_time_ms": round(t_triton, 3),
            "speedup": round(speedup, 2) if speedup != float("inf") else None,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "metadata": {
                "N": N,
                "dtype": str(coords.dtype),
                "device": str(coords.device),
            }
        })

        # Print summary for this case
        print(f"N={N:4d} | torch: {t_torch:6.3f} ms | triton: {t_triton:6.3f} ms | "
              f"speedup: {speedup:.2f}x | max_diff: {max_diff:.2e}")

    return results


def main():
    results = run_benchmark()
    
    print("\n" + "="*80)
    print("Pairwise Distance Benchmark: PyTorch cdist vs Triton")
    print("="*80)
    print(f"{'N':>6} | {'PyTorch (ms)':>12} | {'Triton (ms)':>12} | {'Speedup':>8} | {'Max Diff':>12}")
    print("-"*80)
    
    for result in results:
        meta = result["metadata"]
        speedup_str = f"{result['speedup']:.2f}x" if result['speedup'] else "inf"
        print(f"{meta['N']:6d} | {result['baseline_time_ms']:12.3f} | "
              f"{result['triton_time_ms']:12.3f} | {speedup_str:>8} | "
              f"{result['max_diff']:12.2e}")


if __name__ == "__main__":
    main()
