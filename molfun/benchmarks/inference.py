"""
Inference performance benchmark.

Measures latency, throughput, and peak memory for a model across
different sequence lengths.  Useful for comparing architectures
or quantifying the cost of LoRA vs full fine-tuning.

Usage::

    from molfun.benchmarks.inference import InferenceBenchmark

    bench = InferenceBenchmark(model, device="cuda")
    results = bench.run(seq_lengths=[128, 256, 512, 1024])
    print(results.to_markdown())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class InferenceResult:
    """Timing results for one sequence length."""

    seq_length: int
    mean_ms: float
    std_ms: float
    throughput_samples_per_s: float
    peak_memory_mb: float


@dataclass
class InferenceReport:
    """Collection of ``InferenceResult`` across sequence lengths."""

    model_name: str
    device: str
    results: list[InferenceResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            f"### Inference: {self.model_name} ({self.device})",
            "",
            "| Seq Length | Latency (ms) | Std (ms) | Throughput (samples/s) | Peak VRAM (MB) |",
            "|-----------|-------------|---------|----------------------|---------------|",
        ]
        for r in self.results:
            lines.append(
                f"| {r.seq_length} | {r.mean_ms:.2f} | {r.std_ms:.2f} | "
                f"{r.throughput_samples_per_s:.1f} | {r.peak_memory_mb:.0f} |"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "metadata": self.metadata,
            "results": [
                {
                    "seq_length": r.seq_length,
                    "mean_ms": r.mean_ms,
                    "std_ms": r.std_ms,
                    "throughput_samples_per_s": r.throughput_samples_per_s,
                    "peak_memory_mb": r.peak_memory_mb,
                }
                for r in self.results
            ],
        }


class InferenceBenchmark:
    """
    Benchmark model inference latency and memory.

    Generates synthetic inputs at various sequence lengths, runs
    timed forward passes, and records CUDA memory statistics.
    """

    def __init__(
        self,
        model,
        device: str = "cuda",
        warmup: int = 5,
        repeats: int = 20,
        batch_size: int = 1,
    ) -> None:
        self._model = model
        self._device = device
        self._warmup = warmup
        self._repeats = repeats
        self._batch_size = batch_size

    def run(
        self,
        seq_lengths: list[int] | None = None,
    ) -> InferenceReport:
        """
        Run inference benchmark across sequence lengths.

        Args:
            seq_lengths: List of sequence lengths to test.
                Defaults to ``[64, 128, 256, 512, 1024]``.
        """
        if seq_lengths is None:
            seq_lengths = [64, 128, 256, 512, 1024]

        self._set_eval_mode()
        report = InferenceReport(
            model_name=self._model_name(),
            device=self._device,
            metadata=self._collect_metadata(),
        )

        for L in seq_lengths:
            result = self._benchmark_length(L)
            report.results.append(result)

        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _benchmark_length(self, seq_len: int) -> InferenceResult:
        """Time forward passes for a single sequence length."""
        batch = self._make_synthetic_batch(seq_len)
        use_cuda = "cuda" in self._device and torch.cuda.is_available()

        # Warmup
        with torch.no_grad():
            for _ in range(self._warmup):
                self._model.forward(batch)
        if use_cuda:
            torch.cuda.synchronize()

        # Reset memory tracking
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        # Timed runs
        times_ms: list[float] = []
        with torch.no_grad():
            for _ in range(self._repeats):
                if use_cuda:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    self._model.forward(batch)
                    end.record()
                    torch.cuda.synchronize()
                    times_ms.append(start.elapsed_time(end))
                else:
                    t0 = time.perf_counter()
                    self._model.forward(batch)
                    times_ms.append((time.perf_counter() - t0) * 1000)

        t = torch.tensor(times_ms)
        mean_ms = t.mean().item()
        std_ms = t.std().item() if len(times_ms) > 1 else 0.0
        throughput = (self._batch_size / (mean_ms / 1000)) if mean_ms > 0 else 0.0

        peak_mb = 0.0
        if use_cuda:
            peak_mb = torch.cuda.max_memory_allocated() / 1e6

        return InferenceResult(
            seq_length=seq_len,
            mean_ms=round(mean_ms, 3),
            std_ms=round(std_ms, 3),
            throughput_samples_per_s=round(throughput, 1),
            peak_memory_mb=round(peak_mb, 1),
        )

    def _make_synthetic_batch(self, seq_len: int) -> dict:
        """Generate a minimal synthetic input batch."""
        B = self._batch_size
        return {
            "aatype": torch.randint(0, 20, (B, seq_len), device=self._device),
            "residue_index": torch.arange(seq_len, device=self._device).unsqueeze(0).expand(B, -1),
            "all_atom_positions": torch.randn(B, seq_len, 3, device=self._device),
            "all_atom_mask": torch.ones(B, seq_len, device=self._device),
            "seq_length": torch.tensor([seq_len] * B, device=self._device),
        }

    def _set_eval_mode(self) -> None:
        if hasattr(self._model, "adapter"):
            self._model.adapter.eval()
        if hasattr(self._model, "head") and self._model.head is not None:
            self._model.head.eval()

    def _model_name(self) -> str:
        if hasattr(self._model, "model_type"):
            return str(self._model.model_type)
        return type(self._model).__name__

    def _collect_metadata(self) -> dict:
        meta: dict = {
            "warmup": self._warmup,
            "repeats": self._repeats,
            "batch_size": self._batch_size,
        }
        if torch.cuda.is_available() and "cuda" in self._device:
            meta["gpu"] = torch.cuda.get_device_name(0)
        return meta
