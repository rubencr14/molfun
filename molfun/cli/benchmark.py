"""
CLI command for running Molfun benchmarks.
"""

from __future__ import annotations
from enum import Enum
from typing import Annotated

import typer


class BenchTarget(str, Enum):
    all = "all"
    rmsd = "rmsd"
    contacts = "contacts"
    pairwise = "pairwise"
    gelu = "gelu"
    fused = "fused"


def benchmark(
    target: Annotated[BenchTarget, typer.Argument(help="Benchmark to run.")] = BenchTarget.all,
    sizes: Annotated[str, typer.Option(help="Comma-separated atom/dimension sizes.")] = "256,512,1024,2048",
    device: Annotated[str, typer.Option(help="Device: cuda, cpu.")] = "cuda",
    warmup: Annotated[int, typer.Option(help="Warmup iterations.")] = 5,
    repeats: Annotated[int, typer.Option(help="Benchmark repetitions.")] = 20,
):
    """Run performance benchmarks for Molfun kernels and operations."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        typer.echo("CUDA not available, falling back to CPU.")
        device = "cpu"

    size_list = [int(s.strip()) for s in sizes.split(",")]

    benchmarks = _select_benchmarks(target)

    if not benchmarks:
        typer.echo("No benchmarks available. Install triton for GPU kernels: pip install molfun[kernels]")
        return

    for name, bench_fn in benchmarks:
        typer.echo(f"\n{'═' * 60}")
        typer.echo(f"  {name}")
        typer.echo(f"{'═' * 60}")
        try:
            bench_fn(size_list, device, warmup, repeats)
        except ImportError as e:
            typer.echo(f"  Skipped: {e}")
        except Exception as e:
            typer.echo(f"  Error: {e}")

    typer.echo(f"\n{'═' * 60}")
    typer.echo("  Benchmarks complete.")
    typer.echo(f"{'═' * 60}")


def _select_benchmarks(target: BenchTarget) -> list[tuple[str, callable]]:
    benchmarks = []

    if target in (BenchTarget.all, BenchTarget.rmsd):
        benchmarks.append(("RMSD Kernel", _bench_rmsd))

    if target in (BenchTarget.all, BenchTarget.contacts):
        benchmarks.append(("Contact Map Kernel", _bench_contacts))

    if target in (BenchTarget.all, BenchTarget.pairwise):
        benchmarks.append(("Pairwise Distance Kernel", _bench_pairwise))

    if target in (BenchTarget.all, BenchTarget.gelu):
        benchmarks.append(("GELU Kernel", _bench_gelu))

    if target in (BenchTarget.all, BenchTarget.fused):
        benchmarks.append(("Fused Linear+GELU Kernel", _bench_fused))

    return benchmarks


def _bench_rmsd(sizes, device, warmup, repeats):
    import torch
    from molfun.kernels.analysis.rmsd import rmsd_triton

    for N in sizes:
        coords_a = torch.randn(1, N, 3, device=device)
        coords_b = torch.randn(1, N, 3, device=device)

        for _ in range(warmup):
            rmsd_triton(coords_a, coords_b)
        if device == "cuda":
            torch.cuda.synchronize()

        import time
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            rmsd_triton(coords_a, coords_b)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_ms = sum(times) / len(times) * 1000
        typer.echo(f"  N={N:6d}: {avg_ms:.3f} ms (avg over {repeats} runs)")


def _bench_contacts(sizes, device, warmup, repeats):
    import torch
    from molfun.kernels.analysis.contact_map import contact_map_triton

    for N in sizes:
        coords = torch.randn(1, N, 3, device=device)

        for _ in range(warmup):
            contact_map_triton(coords, threshold=8.0)
        if device == "cuda":
            torch.cuda.synchronize()

        import time
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            contact_map_triton(coords, threshold=8.0)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_ms = sum(times) / len(times) * 1000
        typer.echo(f"  N={N:6d}: {avg_ms:.3f} ms (avg over {repeats} runs)")


def _bench_pairwise(sizes, device, warmup, repeats):
    import torch
    from molfun.kernels.analysis.pairwise_distance import pairwise_distance_triton

    for N in sizes:
        coords = torch.randn(1, N, 3, device=device)

        for _ in range(warmup):
            pairwise_distance_triton(coords)
        if device == "cuda":
            torch.cuda.synchronize()

        import time
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            pairwise_distance_triton(coords)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_ms = sum(times) / len(times) * 1000
        typer.echo(f"  N={N:6d}: {avg_ms:.3f} ms (avg over {repeats} runs)")


def _bench_gelu(sizes, device, warmup, repeats):
    import torch
    from molfun.kernels.models.gelu_kernel import gelu_triton

    for D in sizes:
        x = torch.randn(32, D, device=device)

        for _ in range(warmup):
            gelu_triton(x)
        if device == "cuda":
            torch.cuda.synchronize()

        import time
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            gelu_triton(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_ms = sum(times) / len(times) * 1000
        typer.echo(f"  D={D:6d}: {avg_ms:.3f} ms (avg over {repeats} runs)")


def _bench_fused(sizes, device, warmup, repeats):
    import torch
    from molfun.kernels.models.fused_linear_gelu import fused_linear_gelu

    for D in sizes:
        x = torch.randn(32, D, device=device)
        w = torch.randn(D, D, device=device)
        b = torch.randn(D, device=device)

        for _ in range(warmup):
            fused_linear_gelu(x, w, b)
        if device == "cuda":
            torch.cuda.synchronize()

        import time
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fused_linear_gelu(x, w, b)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_ms = sum(times) / len(times) * 1000
        typer.echo(f"  D={D:6d}: {avg_ms:.3f} ms (avg over {repeats} runs)")
