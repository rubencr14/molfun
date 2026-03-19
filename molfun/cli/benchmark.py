"""
CLI command for running Molfun benchmarks.

Supports kernel benchmarks (Triton timing), model evaluation,
inference profiling, and data pipeline throughput.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer


class BenchCategory(str, Enum):
    kernels = "kernels"
    model = "model"
    inference = "inference"
    training = "training"
    data = "data"


def benchmark(
    category: Annotated[BenchCategory, typer.Argument(help="Benchmark category.")] = BenchCategory.kernels,
    target: Annotated[str, typer.Option(help="Specific target (e.g. rmsd, contacts, all).")] = "all",
    sizes: Annotated[str, typer.Option(help="Comma-separated sizes for kernel benchmarks.")] = "256,512,1024,2048",
    device: Annotated[str, typer.Option(help="Device: cuda, cpu, mps.")] = "cuda",
    warmup: Annotated[int, typer.Option(help="Warmup iterations.")] = 5,
    repeats: Annotated[int, typer.Option(help="Repetitions.")] = 20,
    checkpoint: Annotated[Optional[Path], typer.Option(help="Model checkpoint for model/inference benchmarks.")] = None,
    suite: Annotated[str, typer.Option(help="Benchmark suite name (pdbbind, atom3d_lba, flip, structure).")] = "pdbbind",
    data_dir: Annotated[Optional[Path], typer.Option(help="Data directory for evaluation.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
):
    """Run benchmarks: kernels, model evaluation, inference, training, or data pipeline."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        typer.echo("CUDA not available, falling back to CPU.")
        device = "cpu"

    if category == BenchCategory.kernels:
        _run_kernel_benchmarks(target, sizes, device, warmup, repeats)
    elif category == BenchCategory.model:
        _run_model_benchmark(checkpoint, suite, data_dir, device, json_output)
    elif category == BenchCategory.inference:
        _run_inference_benchmark(checkpoint, device, sizes, warmup, repeats, json_output)
    elif category == BenchCategory.training:
        typer.echo("Training benchmarks require a script.  See examples/custom_architecture.py")
        typer.echo("or use molfun.benchmarks.StrategyComparison programmatically.")
    elif category == BenchCategory.data:
        _run_data_benchmark(data_dir, repeats, json_output)


# ------------------------------------------------------------------
# Kernel benchmarks (legacy + reorganised)
# ------------------------------------------------------------------

_KERNEL_TARGETS = ("rmsd", "contacts", "pairwise", "gelu", "fused")


def _run_kernel_benchmarks(target: str, sizes: str, device: str, warmup: int, repeats: int):
    size_list = [int(s.strip()) for s in sizes.split(",")]
    targets = _KERNEL_TARGETS if target == "all" else (target,)

    for name in targets:
        bench_fn = _KERNEL_FNS.get(name)
        if bench_fn is None:
            typer.echo(f"Unknown kernel target: {name}")
            continue
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"  {name.upper()}")
        typer.echo(f"{'=' * 60}")
        try:
            bench_fn(size_list, device, warmup, repeats)
        except ImportError as e:
            typer.echo(f"  Skipped: {e}")
        except Exception as e:
            typer.echo(f"  Error: {e}")

    typer.echo(f"\n{'=' * 60}\n  Kernel benchmarks complete.\n{'=' * 60}")


def _bench_rmsd(sizes, device, warmup, repeats):
    import torch
    from molfun.kernels.analysis.rmsd import rmsd_triton
    from molfun.benchmarks.kernels.timing import time_it_cuda, time_it_cpu

    for N in sizes:
        a = torch.randn(1, N, 3, device=device)
        b = torch.randn(1, N, 3, device=device)
        timer = time_it_cuda if device == "cuda" else time_it_cpu
        ms = timer(lambda: rmsd_triton(a, b), iters=repeats, warmup=warmup)
        typer.echo(f"  N={N:6d}: {ms:.3f} ms")


def _bench_contacts(sizes, device, warmup, repeats):
    import torch
    from molfun.kernels.analysis.contact_map import contact_map_triton
    from molfun.benchmarks.kernels.timing import time_it_cuda, time_it_cpu

    for N in sizes:
        coords = torch.randn(1, N, 3, device=device)
        timer = time_it_cuda if device == "cuda" else time_it_cpu
        ms = timer(lambda: contact_map_triton(coords, threshold=8.0), iters=repeats, warmup=warmup)
        typer.echo(f"  N={N:6d}: {ms:.3f} ms")


def _bench_pairwise(sizes, device, warmup, repeats):
    import torch
    from molfun.kernels.analysis.pairwise_distance import pairwise_distance_triton
    from molfun.benchmarks.kernels.timing import time_it_cuda, time_it_cpu

    for N in sizes:
        coords = torch.randn(1, N, 3, device=device)
        timer = time_it_cuda if device == "cuda" else time_it_cpu
        ms = timer(lambda: pairwise_distance_triton(coords), iters=repeats, warmup=warmup)
        typer.echo(f"  N={N:6d}: {ms:.3f} ms")


def _bench_gelu(sizes, device, warmup, repeats):
    import torch
    from molfun.kernels.models.gelu_kernel import gelu_triton
    from molfun.benchmarks.kernels.timing import time_it_cuda, time_it_cpu

    for D in sizes:
        x = torch.randn(32, D, device=device)
        timer = time_it_cuda if device == "cuda" else time_it_cpu
        ms = timer(lambda: gelu_triton(x), iters=repeats, warmup=warmup)
        typer.echo(f"  D={D:6d}: {ms:.3f} ms")


def _bench_fused(sizes, device, warmup, repeats):
    import torch
    from molfun.kernels.models.fused_linear_gelu import fused_linear_gelu
    from molfun.benchmarks.kernels.timing import time_it_cuda, time_it_cpu

    for D in sizes:
        x = torch.randn(32, D, device=device)
        w = torch.randn(D, D, device=device)
        b = torch.randn(D, device=device)
        timer = time_it_cuda if device == "cuda" else time_it_cpu
        ms = timer(lambda: fused_linear_gelu(x, w, b), iters=repeats, warmup=warmup)
        typer.echo(f"  D={D:6d}: {ms:.3f} ms")


_KERNEL_FNS = {
    "rmsd": _bench_rmsd,
    "contacts": _bench_contacts,
    "pairwise": _bench_pairwise,
    "gelu": _bench_gelu,
    "fused": _bench_fused,
}


# ------------------------------------------------------------------
# Model evaluation benchmark
# ------------------------------------------------------------------

def _run_model_benchmark(checkpoint, suite_name, data_dir, device, json_output):
    import json

    if checkpoint is None:
        typer.echo("Error: --checkpoint is required for model benchmarks.")
        raise typer.Exit(1)

    from molfun.models.structure import MolfunStructureModel
    from molfun.benchmarks import ModelEvaluator, BenchmarkSuite

    typer.echo(f"Loading model from {checkpoint}...")
    model = MolfunStructureModel.load(str(checkpoint), device=device)

    suite_map = {
        "pdbbind": BenchmarkSuite.pdbbind,
        "atom3d_lba": BenchmarkSuite.atom3d_lba,
        "atom3d_psr": BenchmarkSuite.atom3d_psr,
        "flip": BenchmarkSuite.flip,
        "structure": BenchmarkSuite.structure_quality,
    }

    factory = suite_map.get(suite_name)
    if factory is None:
        typer.echo(f"Unknown suite: {suite_name}. Available: {list(suite_map)}")
        raise typer.Exit(1)

    kwargs = {}
    if data_dir and factory != BenchmarkSuite.flip:
        kwargs["structures_dir" if "pdbbind" in suite_name else "data_dir"] = str(data_dir)

    suite = factory(**kwargs) if kwargs else factory()
    typer.echo(f"Running suite: {suite.name} ({len(suite)} tasks)")

    evaluator = ModelEvaluator(model, suite, device=device)
    report = evaluator.run()

    if json_output:
        typer.echo(report.to_json())
    else:
        typer.echo(report.to_markdown())


# ------------------------------------------------------------------
# Inference benchmark
# ------------------------------------------------------------------

def _run_inference_benchmark(checkpoint, device, sizes, warmup, repeats, json_output):
    import json

    if checkpoint is None:
        typer.echo("Error: --checkpoint is required for inference benchmarks.")
        raise typer.Exit(1)

    from molfun.models.structure import MolfunStructureModel
    from molfun.benchmarks.inference import InferenceBenchmark

    model = MolfunStructureModel.load(str(checkpoint), device=device)
    seq_lengths = [int(s.strip()) for s in sizes.split(",")]

    bench = InferenceBenchmark(model, device=device, warmup=warmup, repeats=repeats)
    report = bench.run(seq_lengths=seq_lengths)

    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        typer.echo(report.to_markdown())


# ------------------------------------------------------------------
# Data pipeline benchmark
# ------------------------------------------------------------------

def _run_data_benchmark(data_dir, repeats, json_output):
    import json

    if data_dir is None:
        typer.echo("Error: --data-dir is required for data benchmarks.")
        raise typer.Exit(1)

    from molfun.benchmarks.data_pipeline import ParsingBenchmark

    files = []
    for ext in ("*.pdb", "*.cif", "*.sdf", "*.mol2", "*.a3m", "*.fasta"):
        files.extend(str(p) for p in Path(data_dir).glob(ext))

    if not files:
        typer.echo(f"No parseable files found in {data_dir}")
        raise typer.Exit(1)

    typer.echo(f"Benchmarking {len(files)} files...")
    bench = ParsingBenchmark(repeats=repeats)
    report = bench.run(files)

    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        typer.echo(report.to_markdown())
