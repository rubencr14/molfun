"""
Data pipeline benchmarks.

``ParsingBenchmark`` measures the throughput of each parser across formats.
``LoadingBenchmark`` measures DataLoader throughput (local vs streaming,
single-worker vs multi-worker).

Usage::

    from molfun.benchmarks.data_pipeline import ParsingBenchmark

    bench = ParsingBenchmark()
    report = bench.run(["data/structures/1a2b.pdb", "data/structures/7bv2.cif"])
    print(report.to_markdown())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader


# ------------------------------------------------------------------
# Parsing benchmark
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ParsingResult:
    """Throughput for one parser / file combination."""

    parser_name: str
    file_path: str
    file_size_kb: float
    parse_time_ms: float
    throughput_files_per_s: float
    success: bool
    error: str = ""


@dataclass
class ParsingReport:
    """Aggregate parsing benchmark results."""

    results: list[ParsingResult] = field(default_factory=list)

    def to_markdown(self) -> str:
        if not self.results:
            return "*No results*"

        lines = [
            "### Parsing Throughput",
            "",
            "| Parser | File | Size (KB) | Time (ms) | Files/s | OK |",
            "|--------|------|----------|----------|---------|-----|",
        ]
        for r in self.results:
            ok = "yes" if r.success else f"no: {r.error}"
            lines.append(
                f"| {r.parser_name} | {Path(r.file_path).name} | "
                f"{r.file_size_kb:.1f} | {r.parse_time_ms:.2f} | "
                f"{r.throughput_files_per_s:.1f} | {ok} |"
            )
        return "\n".join(lines)

    def to_dict(self) -> list[dict]:
        return [
            {
                "parser": r.parser_name,
                "file": r.file_path,
                "size_kb": r.file_size_kb,
                "time_ms": r.parse_time_ms,
                "throughput": r.throughput_files_per_s,
                "success": r.success,
                "error": r.error,
            }
            for r in self.results
        ]

    def mean_throughput(self) -> float:
        """Average files/s across all successful parses."""
        ok = [r for r in self.results if r.success]
        if not ok:
            return 0.0
        return sum(r.throughput_files_per_s for r in ok) / len(ok)


class ParsingBenchmark:
    """
    Benchmark parser throughput across file formats.

    Automatically selects the appropriate parser for each file using
    ``auto_parser`` from the parser registry.
    """

    def __init__(self, repeats: int = 10, warmup: int = 2) -> None:
        self._repeats = repeats
        self._warmup = warmup

    def run(self, file_paths: list[str]) -> ParsingReport:
        """
        Parse each file ``repeats`` times and measure throughput.

        Args:
            file_paths: Paths to structure/ligand/alignment files.
        """
        from molfun.data.parsers import auto_parser

        report = ParsingReport()

        for path_str in file_paths:
            p = Path(path_str)
            if not p.exists():
                report.results.append(ParsingResult(
                    parser_name="unknown",
                    file_path=path_str,
                    file_size_kb=0.0,
                    parse_time_ms=0.0,
                    throughput_files_per_s=0.0,
                    success=False,
                    error="File not found",
                ))
                continue

            size_kb = p.stat().st_size / 1024
            try:
                parser = auto_parser(path_str)
            except ValueError as e:
                report.results.append(ParsingResult(
                    parser_name="unknown",
                    file_path=path_str,
                    file_size_kb=size_kb,
                    parse_time_ms=0.0,
                    throughput_files_per_s=0.0,
                    success=False,
                    error=str(e),
                ))
                continue

            parser_name = type(parser).__name__

            parse_fn = getattr(parser, "parse_file", None) or getattr(parser, "parse", None)
            if parse_fn is None:
                report.results.append(ParsingResult(
                    parser_name=parser_name,
                    file_path=path_str,
                    file_size_kb=size_kb,
                    parse_time_ms=0.0,
                    throughput_files_per_s=0.0,
                    success=False,
                    error="Parser has no parse_file or parse method",
                ))
                continue

            # Warmup
            for _ in range(self._warmup):
                try:
                    parse_fn(path_str)
                except Exception:
                    break

            # Timed runs
            times_ms: list[float] = []
            error = ""
            success = True
            for _ in range(self._repeats):
                t0 = time.perf_counter()
                try:
                    parse_fn(path_str)
                except Exception as e:
                    error = str(e)
                    success = False
                    break
                times_ms.append((time.perf_counter() - t0) * 1000)

            if times_ms:
                mean_ms = sum(times_ms) / len(times_ms)
                throughput = 1000.0 / mean_ms if mean_ms > 0 else 0.0
            else:
                mean_ms = 0.0
                throughput = 0.0

            report.results.append(ParsingResult(
                parser_name=parser_name,
                file_path=path_str,
                file_size_kb=size_kb,
                parse_time_ms=round(mean_ms, 3),
                throughput_files_per_s=round(throughput, 1),
                success=success,
                error=error,
            ))

        return report


# ------------------------------------------------------------------
# DataLoader benchmark
# ------------------------------------------------------------------

@dataclass(frozen=True)
class LoadingResult:
    """Throughput for one DataLoader configuration."""

    name: str
    num_workers: int
    batch_size: int
    total_samples: int
    total_time_s: float
    samples_per_s: float
    batches_per_s: float


@dataclass
class LoadingReport:
    """Aggregate DataLoader benchmark results."""

    results: list[LoadingResult] = field(default_factory=list)

    def to_markdown(self) -> str:
        if not self.results:
            return "*No results*"

        lines = [
            "### DataLoader Throughput",
            "",
            "| Config | Workers | Batch | Samples | Time (s) | Samples/s | Batches/s |",
            "|--------|---------|-------|---------|---------|----------|----------|",
        ]
        for r in self.results:
            lines.append(
                f"| {r.name} | {r.num_workers} | {r.batch_size} | "
                f"{r.total_samples} | {r.total_time_s:.2f} | "
                f"{r.samples_per_s:.1f} | {r.batches_per_s:.1f} |"
            )
        return "\n".join(lines)


class LoadingBenchmark:
    """
    Benchmark DataLoader throughput across configurations.

    Tests different ``num_workers`` and ``batch_size`` settings
    to find optimal data loading parameters.
    """

    def __init__(
        self,
        dataset,
        collate_fn=None,
        max_batches: int = 50,
    ) -> None:
        self._dataset = dataset
        self._collate_fn = collate_fn
        self._max_batches = max_batches

    def run(
        self,
        worker_counts: list[int] | None = None,
        batch_sizes: list[int] | None = None,
    ) -> LoadingReport:
        """
        Measure throughput for each (workers, batch_size) combination.

        Args:
            worker_counts: Number of DataLoader workers to test.
            batch_sizes: Batch sizes to test.
        """
        if worker_counts is None:
            worker_counts = [0, 2, 4]
        if batch_sizes is None:
            batch_sizes = [1]

        report = LoadingReport()

        for nw in worker_counts:
            for bs in batch_sizes:
                result = self._benchmark_config(nw, bs)
                report.results.append(result)

        return report

    def _benchmark_config(self, num_workers: int, batch_size: int) -> LoadingResult:
        """Time a single DataLoader configuration."""
        loader = DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )

        n_batches = 0
        n_samples = 0
        t0 = time.perf_counter()

        for batch in loader:
            n_batches += 1
            if isinstance(batch, (list, tuple)):
                n_samples += len(batch[0]) if hasattr(batch[0], "__len__") else batch_size
            elif isinstance(batch, dict):
                first_val = next(iter(batch.values()))
                n_samples += first_val.shape[0] if isinstance(first_val, torch.Tensor) else batch_size
            else:
                n_samples += batch_size
            if n_batches >= self._max_batches:
                break

        total_time = time.perf_counter() - t0

        return LoadingResult(
            name=f"w{num_workers}_b{batch_size}",
            num_workers=num_workers,
            batch_size=batch_size,
            total_samples=n_samples,
            total_time_s=round(total_time, 3),
            samples_per_s=round(n_samples / total_time, 1) if total_time > 0 else 0.0,
            batches_per_s=round(n_batches / total_time, 1) if total_time > 0 else 0.0,
        )
