import time
import torch
import torch.nn.functional as F
from typing import List, Dict, Any

from molfun.kernels.models.gelu_triton import gelu_triton


def time_it(fn, iters=200, warmup=50):
    # Warmup (para JIT / caches / clocks)
    for _ in range(warmup):
        y = fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / iters


def run_benchmark() -> List[Dict[str, Any]]:
    """Ejecuta el benchmark y devuelve resultados estructurados"""
    torch.set_grad_enabled(False)
    assert torch.cuda.is_available()

    # Shapes típicos para ESM-2 pequeño:
    # esm2_t6_8M => hidden size ~320, y el MLP intermedio suele ser ~4x => ~1280
    shapes = [
        (2, 256, 1280),
        (2, 512, 1280),
        (1, 1024, 1280),
        (1, 2048, 1280),
    ]

    results = []

    for B, T, D in shapes:
        x = torch.randn(B, T, D, device="cuda", dtype=torch.float16)

        # Importante: primera llamada a gelu_triton compila (JIT).
        # Para que el bench sea justo, "calienta" UNA vez.
        _ = gelu_triton(x)
        torch.cuda.synchronize()

        # Baseline PyTorch GELU
        t_torch = time_it(lambda: F.gelu(x), iters=200, warmup=50)

        # Triton GELU
        t_triton = time_it(lambda: gelu_triton(x), iters=200, warmup=50)

        # Correctness check (tolerancias para fp16)
        y_ref = F.gelu(x)
        y_tri = gelu_triton(x)
        max_abs = (y_ref - y_tri).abs().max().item()
        mean_abs = (y_ref - y_tri).abs().mean().item()

        speedup = t_torch / t_triton if t_triton > 0 else float("inf")

        results.append({
            "benchmark_name": "gelu",
            "case_name": f"B={B}_T={T}_D={D}",
            "baseline_time_ms": round(t_torch * 1e3, 3),
            "triton_time_ms": round(t_triton * 1e3, 3),
            "speedup": round(speedup, 2) if speedup != float("inf") else None,
            "max_diff": max_abs,
            "mean_diff": mean_abs,
            "metadata": {
                "B": B,
                "T": T,
                "D": D,
                "dtype": "float16"
            }
        })

    return results


def main():
    results = run_benchmark()
    for result in results:
        meta = result["metadata"]
        print(f"\nShape B={meta['B']} T={meta['T']} D={meta['D']}")
        print(f"  torch.gelu:  {result['baseline_time_ms']:.3f} ms")
        print(f"  triton.gelu: {result['triton_time_ms']:.3f} ms")
        print(f"  speedup:     {result['speedup']:.2f}x" if result['speedup'] else "  speedup:     inf")
        print(f"  max|diff|:   {result['max_diff']:.3e}, mean|diff|: {result['mean_diff']:.3e}")


if __name__ == "__main__":
    main()
