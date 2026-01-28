import torch
import torch.nn.functional as F
from typing import List, Dict, Any

from src.kernels.fused_linear_gelu_triton import fused_linear_gelu_triton


def time_it_cuda(fn, iters=200, warmup=50):
    # Warmup (incluye autotune / caches / clocks)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    return ms / iters


def run_benchmark() -> List[Dict[str, Any]]:
    """Ejecuta el benchmark y devuelve resultados estructurados"""
    torch.set_grad_enabled(False)
    assert torch.cuda.is_available()

    # Shapes inspirados en ESM-2 t6 (D=320, 4D=1280)
    # M = B*T (tokens)
    cases = [
        (2, 256, 320, 1280),
        (2, 512, 320, 1280),
        (1, 1024, 320, 1280),
        (1, 2048, 320, 1280),
        # caso más "cargado" para que se note más
        (4, 2048, 320, 1280),
        (16, 2048, 320, 1280),
        (16, 2048, 320, 1280),
        (32, 2048, 320, 1280),
        (64, 2048, 320, 1280),
        (128, 2048, 320, 1280),
        (256, 2048, 320, 1280),
        (512, 2048, 320, 1280),
    ]

    results = []

    for B, T, K, N in cases:
        x = torch.randn(B, T, K, device="cuda", dtype=torch.float16)
        w = torch.randn(N, K, device="cuda", dtype=torch.float16)  # nn.Linear layout
        b = torch.randn(N, device="cuda", dtype=torch.float16)

        # Force one run to trigger JIT + autotune before timing
        _ = fused_linear_gelu_triton(x, w, b)
        torch.cuda.synchronize()

        def baseline():
            y = F.linear(x, w, b)
            return F.gelu(y, approximate="none")

        def triton_fused():
            return fused_linear_gelu_triton(x, w, b)

        t_base = time_it_cuda(baseline, iters=200, warmup=50)
        t_tri = time_it_cuda(triton_fused, iters=200, warmup=50)

        # correctness
        y_ref = baseline()
        y_tri = triton_fused()
        max_abs = (y_ref - y_tri).abs().max().item()
        mean_abs = (y_ref - y_tri).abs().mean().item()

        speedup = t_base / t_tri if t_tri > 0 else float("inf")

        results.append({
            "benchmark_name": "fused_linear_gelu",
            "case_name": f"B={B}_T={T}_K={K}_N={N}",
            "baseline_time_ms": round(t_base, 4),
            "triton_time_ms": round(t_tri, 4),
            "speedup": round(speedup, 2) if speedup != float("inf") else None,
            "max_diff": max_abs,
            "mean_diff": mean_abs,
            "metadata": {
                "B": B,
                "T": T,
                "K": K,
                "N": N,
                "M": B * T,
                "dtype": "float16"
            }
        })

    return results


def main():
    results = run_benchmark()
    for result in results:
        meta = result["metadata"]
        print(f"\nB={meta['B']} T={meta['T']} K={meta['K']} N={meta['N']} (M={meta['M']})")
        print(f"  baseline (linear+gelu): {result['baseline_time_ms']:.4f} ms")
        print(f"  triton fused:           {result['triton_time_ms']:.4f} ms")
        print(f"  speedup:                {result['speedup']:.2f}x" if result['speedup'] else "  speedup:                inf")
        print(f"  max|diff|:              {result['max_diff']:.3e}  mean|diff|: {result['mean_diff']:.3e}")


if __name__ == "__main__":
    main()
