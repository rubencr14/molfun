"""
Training and fine-tuning benchmarks.

``ConvergenceBenchmark`` measures how fast a model reaches a target loss.
``StrategyComparison`` compares multiple fine-tuning strategies (LoRA,
Partial, Full) on the same data, recording VRAM, throughput, and final
quality.

Usage::

    from molfun.benchmarks.training import StrategyComparison

    comp = StrategyComparison(model_factory, train_loader, val_loader)
    comp.add("lora_r4", LoRAFinetune(rank=4, lr_lora=1e-4))
    comp.add("lora_r16", LoRAFinetune(rank=16, lr_lora=1e-4))
    comp.add("partial", PartialFinetune(lr=1e-4))
    report = comp.run(epochs=10)
    print(report.to_markdown())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainingResult:
    """Metrics from a single training run."""

    name: str
    strategy_type: str
    final_train_loss: float
    best_val_loss: float
    epochs_to_best: int
    total_epochs: int
    total_time_s: float
    steps_per_second: float
    peak_memory_mb: float
    trainable_params: int
    total_params: int
    history: list[dict] = field(default_factory=list)


@dataclass
class TrainingReport:
    """Comparison report across multiple training runs."""

    results: list[TrainingResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        if not self.results:
            return "*No results*"

        lines = [
            "### Fine-tuning Strategy Comparison",
            "",
            "| Strategy | Trainable % | Best Val Loss | Epochs | Time (s) | Steps/s | Peak VRAM (MB) |",
            "|----------|------------|--------------|--------|---------|---------|---------------|",
        ]
        for r in self.results:
            pct = r.trainable_params / max(r.total_params, 1) * 100
            lines.append(
                f"| {r.name} | {pct:.1f}% | {r.best_val_loss:.4f} | "
                f"{r.epochs_to_best}/{r.total_epochs} | {r.total_time_s:.1f} | "
                f"{r.steps_per_second:.1f} | {r.peak_memory_mb:.0f} |"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "results": [
                {
                    "name": r.name,
                    "strategy_type": r.strategy_type,
                    "final_train_loss": r.final_train_loss,
                    "best_val_loss": r.best_val_loss,
                    "epochs_to_best": r.epochs_to_best,
                    "total_epochs": r.total_epochs,
                    "total_time_s": r.total_time_s,
                    "steps_per_second": r.steps_per_second,
                    "peak_memory_mb": r.peak_memory_mb,
                    "trainable_params": r.trainable_params,
                    "total_params": r.total_params,
                }
                for r in self.results
            ],
        }


class ConvergenceBenchmark:
    """
    Measure how fast a model reaches a target validation loss.

    Trains until ``target_loss`` is reached or ``max_epochs`` is hit,
    recording time and epoch count.
    """

    def __init__(
        self,
        model,
        strategy,
        train_loader: DataLoader,
        val_loader: DataLoader,
        target_loss: float = 0.5,
        max_epochs: int = 50,
        device: str = "cuda",
    ) -> None:
        self._model = model
        self._strategy = strategy
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._target_loss = target_loss
        self._max_epochs = max_epochs
        self._device = device

    def run(self) -> TrainingResult:
        """Train and measure convergence."""
        use_cuda = "cuda" in self._device and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        history = self._strategy.fit(
            self._model, self._train_loader, self._val_loader,
            epochs=self._max_epochs,
        )
        total_time = time.perf_counter() - t0

        val_losses = [h.get("val_loss", float("inf")) for h in history]
        best_val = min(val_losses) if val_losses else float("inf")
        epochs_to_best = val_losses.index(best_val) + 1 if val_losses else 0

        total_steps = sum(h.get("n_batches", len(self._train_loader)) for h in history)
        steps_per_s = total_steps / total_time if total_time > 0 else 0.0

        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if use_cuda else 0.0

        trainable, total = _count_params(self._model)

        return TrainingResult(
            name=type(self._strategy).__name__,
            strategy_type=type(self._strategy).__name__,
            final_train_loss=history[-1].get("train_loss", 0.0) if history else 0.0,
            best_val_loss=best_val,
            epochs_to_best=epochs_to_best,
            total_epochs=len(history),
            total_time_s=round(total_time, 2),
            steps_per_second=round(steps_per_s, 2),
            peak_memory_mb=round(peak_mb, 1),
            trainable_params=trainable,
            total_params=total,
            history=history,
        )


class StrategyComparison:
    """
    Compare multiple fine-tuning strategies on the same data.

    Uses a ``model_factory`` to create a fresh model for each strategy,
    ensuring fair comparison.

    Usage::

        comp = StrategyComparison(make_model, train_loader, val_loader)
        comp.add("lora_r8", LoRAFinetune(rank=8))
        comp.add("full", FullFinetune(lr=1e-4))
        report = comp.run(epochs=10)
    """

    def __init__(
        self,
        model_factory: Callable,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
    ) -> None:
        self._model_factory = model_factory
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device = device
        self._entries: list[tuple[str, object]] = []

    def add(self, name: str, strategy) -> StrategyComparison:
        """Register a named strategy for comparison.  Returns ``self`` for chaining."""
        self._entries.append((name, strategy))
        return self

    def run(self, epochs: int = 10) -> TrainingReport:
        """
        Train each strategy on a fresh model and collect results.
        """
        report = TrainingReport(metadata={"device": self._device, "epochs": epochs})

        for name, strategy in self._entries:
            model = self._model_factory()
            bench = ConvergenceBenchmark(
                model=model,
                strategy=strategy,
                train_loader=self._train_loader,
                val_loader=self._val_loader,
                max_epochs=epochs,
                device=self._device,
            )
            result = bench.run()
            result.name = name
            report.results.append(result)

        return report


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------

def _count_params(model) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    if hasattr(model, "adapter"):
        module = model.adapter
    elif hasattr(model, "parameters"):
        module = model
    else:
        return 0, 0

    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

    if hasattr(model, "head") and model.head is not None:
        total += sum(p.numel() for p in model.head.parameters())
        trainable += sum(p.numel() for p in model.head.parameters() if p.requires_grad)

    return trainable, total
