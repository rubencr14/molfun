"""
Model evaluator — runs a ``BenchmarkSuite`` against a model.

This is the central orchestrator that ties metrics, suites, and reports
together.  It follows the Template Method pattern: the evaluation loop
is fixed, but metric collection and data loading are delegated.

Usage::

    from molfun.benchmarks import ModelEvaluator, BenchmarkSuite

    suite = BenchmarkSuite.pdbbind()
    evaluator = ModelEvaluator(model, suite, device="cuda")
    report = evaluator.run(tracker=my_tracker)
    print(report.to_markdown())
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import torch
from torch.utils.data import DataLoader

from molfun.benchmarks.metrics import MetricCollection, create_metrics
from molfun.benchmarks.report import BenchmarkReport, TaskResult
from molfun.benchmarks.suites import BenchmarkSuite, BenchmarkTask, TaskType

if TYPE_CHECKING:
    from molfun.tracking.base import BaseTracker


class ModelEvaluator:
    """
    Evaluate a model on every task in a ``BenchmarkSuite``.

    The evaluator:
      1. Iterates over tasks in the suite.
      2. Builds a data loader for each task.
      3. Runs inference and feeds predictions to a ``MetricCollection``.
      4. Collects results into a ``BenchmarkReport``.
      5. Optionally logs to a ``BaseTracker``.

    Subclass and override ``_build_loader`` for custom data pipelines.
    """

    def __init__(
        self,
        model,
        suite: BenchmarkSuite,
        device: str = "cuda",
        batch_size: int = 1,
        num_workers: int = 0,
        max_seq_len: int = 512,
    ) -> None:
        self._model = model
        self._suite = suite
        self._device = device
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._max_seq_len = max_seq_len

    def run(self, tracker: Optional[BaseTracker] = None) -> BenchmarkReport:
        """
        Execute all tasks and return a ``BenchmarkReport``.

        Args:
            tracker: Optional experiment tracker for live logging.
        """
        report = BenchmarkReport(
            model_name=self._model_name(),
            suite_name=self._suite.name,
            metadata=self._collect_metadata(),
        )

        t_total = time.perf_counter()

        for task in self._suite:
            task_result = self._evaluate_task(task, tracker)
            report.results[task.name] = task_result

        report.total_duration_s = time.perf_counter() - t_total

        if tracker is not None:
            flat: dict[str, float] = {}
            for tname, tr in report.results.items():
                for mname, val in tr.metrics.items():
                    flat[f"eval/{tname}/{mname}"] = val
            tracker.log_metrics(flat)

        return report

    # ------------------------------------------------------------------
    # Extension points (Template Method)
    # ------------------------------------------------------------------

    def _build_loader(self, task: BenchmarkTask) -> Optional[DataLoader]:
        """
        Build a DataLoader for a benchmark task.

        Override this method to support custom data formats.
        The default implementation handles common Molfun dataset patterns.
        Returns ``None`` if the data source is not found.
        """
        from pathlib import Path

        source = Path(task.data_source)
        if not source.exists():
            return None

        if task.task_type in (TaskType.REGRESSION, TaskType.CLASSIFICATION):
            return self._loader_from_structures(source, task)
        elif task.task_type == TaskType.STRUCTURE:
            return self._loader_from_structures(source, task)

        return None

    def _extract_predictions(self, result: dict, task: BenchmarkTask) -> Optional[torch.Tensor]:
        """
        Extract the relevant prediction tensor from model output.

        Override for models with non-standard output formats.
        """
        if task.task_type in (TaskType.REGRESSION, TaskType.CLASSIFICATION):
            return result.get("preds")
        elif task.task_type in (TaskType.STRUCTURE, TaskType.DOCKING):
            trunk = result.get("trunk_output")
            if trunk is not None and hasattr(trunk, "positions"):
                return trunk.positions
            return result.get("preds")
        return result.get("preds")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate_task(
        self,
        task: BenchmarkTask,
        tracker: Optional[BaseTracker],
    ) -> TaskResult:
        """Run a single benchmark task."""
        metrics = create_metrics(list(task.metrics))
        loader = self._build_loader(task)

        if loader is None:
            return TaskResult(
                task_name=task.name,
                metrics={m: float("nan") for m in task.metrics},
                n_samples=0,
            )

        self._model_eval_mode()
        n_samples = 0
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch_data in loader:
                batch, targets, mask = self._unpack(batch_data)
                batch = _to_device(batch, self._device)

                result = self._model.forward(batch, mask=mask)
                preds = self._extract_predictions(result, task)

                if preds is not None and targets is not None:
                    preds_cpu = preds.detach().cpu()
                    targets_cpu = targets.detach().cpu()
                    ctx = {}
                    if mask is not None:
                        ctx["mask"] = mask.detach().cpu()
                    metrics.update(preds_cpu, targets_cpu, **ctx)
                    n_samples += preds_cpu.shape[0] if preds_cpu.dim() > 0 else 1

                if task.max_samples and n_samples >= task.max_samples:
                    break

        duration = time.perf_counter() - t0
        computed = metrics.compute()

        return TaskResult(
            task_name=task.name,
            metrics=computed,
            n_samples=n_samples,
            duration_s=round(duration, 2),
        )

    def _model_eval_mode(self) -> None:
        """Set model (and head if present) to eval mode."""
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
            "device": self._device,
            "batch_size": self._batch_size,
            "max_seq_len": self._max_seq_len,
        }
        if torch.cuda.is_available() and "cuda" in self._device:
            meta["gpu"] = torch.cuda.get_device_name(0)
            meta["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
        if hasattr(self._model, "summary"):
            try:
                meta["model_summary"] = self._model.summary()
            except Exception:
                pass
        return meta

    def _loader_from_structures(self, source, task: BenchmarkTask) -> Optional[DataLoader]:
        """Try to build a loader from a directory of PDB/CIF files or a CSV."""
        from pathlib import Path

        source = Path(source)

        if source.is_file() and source.suffix == ".csv":
            try:
                from molfun.data.datasets.structure import StructureDataset, collate_structure_batch
                from molfun.data.sources.affinity import AffinitySource

                affinity = AffinitySource.from_csv(str(source))
                pdb_paths = list(affinity.pdb_paths.values())
                labels = affinity.labels
                ds = StructureDataset(pdb_paths=pdb_paths, labels=labels, max_seq_len=self._max_seq_len)
                return DataLoader(ds, batch_size=self._batch_size, shuffle=False,
                                  num_workers=self._num_workers, collate_fn=collate_structure_batch)
            except Exception:
                return None

        if source.is_dir():
            pdb_files = sorted(source.glob("*.pdb")) + sorted(source.glob("*.cif"))
            if not pdb_files:
                return None
            try:
                from molfun.data.datasets.structure import StructureDataset, collate_structure_batch

                ds = StructureDataset(pdb_paths=pdb_files, max_seq_len=self._max_seq_len)
                return DataLoader(ds, batch_size=self._batch_size, shuffle=False,
                                  num_workers=self._num_workers, collate_fn=collate_structure_batch)
            except Exception:
                return None

        return None

    @staticmethod
    def _unpack(batch_data):
        """Unpack batch into (features, targets, mask) triple."""
        try:
            from molfun.helpers.training import unpack_batch
            return unpack_batch(batch_data)
        except ImportError:
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                return batch_data[0], batch_data[1], batch_data[2] if len(batch_data) > 2 else None
            return batch_data, None, None


def _to_device(data, device: str):
    """Recursively move tensors/dicts to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_to_device(v, device) for v in data)
    return data
