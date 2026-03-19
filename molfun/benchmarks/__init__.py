"""
Molfun benchmarks — kernel performance, model evaluation, and data pipeline profiling.

Subpackages:
    - ``kernels/``: Low-level GPU kernel timing (Triton vs baselines).
    - ``metrics``: Pluggable metric system (MAE, Pearson, TM-score, lDDT, …).
    - ``suites``: Declarative benchmark definitions (PDBbind, ATOM3D, FLIP, …).
    - ``evaluator``: Run a suite against a model and produce a report.
    - ``report``: Export results as Markdown, LaTeX, JSON, or pandas.
    - ``inference``: Latency / throughput / VRAM profiling.
    - ``training``: Fine-tuning convergence and strategy comparison.
    - ``data_pipeline``: Parser and DataLoader throughput.
"""

from molfun.benchmarks.data_pipeline import (
    LoadingBenchmark,
    LoadingReport,
    ParsingBenchmark,
    ParsingReport,
)
from molfun.benchmarks.evaluator import ModelEvaluator
from molfun.benchmarks.inference import InferenceBenchmark, InferenceReport
from molfun.benchmarks.metrics import (
    AUPRC,
    AUROC,
    GDT_TS,
    LDDT,
    MAE,
    METRIC_REGISTRY,
    R2,
    RMSE,
    BaseMetric,
    CoordRMSD,
    DockingSuccess,
    MetricCollection,
    PearsonR,
    SpearmanRho,
    TM_Score,
    create_metrics,
)
from molfun.benchmarks.report import BenchmarkReport, Leaderboard, TaskResult
from molfun.benchmarks.suites import BenchmarkSuite, BenchmarkTask, TaskType
from molfun.benchmarks.training import (
    ConvergenceBenchmark,
    StrategyComparison,
    TrainingReport,
    TrainingResult,
)

__all__ = [
    # Metrics
    "BaseMetric",
    "MAE",
    "RMSE",
    "PearsonR",
    "SpearmanRho",
    "R2",
    "AUROC",
    "AUPRC",
    "CoordRMSD",
    "GDT_TS",
    "LDDT",
    "TM_Score",
    "DockingSuccess",
    "MetricCollection",
    "METRIC_REGISTRY",
    "create_metrics",
    # Suites
    "BenchmarkTask",
    "BenchmarkSuite",
    "TaskType",
    # Evaluator
    "ModelEvaluator",
    # Reports
    "BenchmarkReport",
    "TaskResult",
    "Leaderboard",
    # Inference
    "InferenceBenchmark",
    "InferenceReport",
    # Training
    "ConvergenceBenchmark",
    "StrategyComparison",
    "TrainingReport",
    "TrainingResult",
    # Data pipeline
    "ParsingBenchmark",
    "LoadingBenchmark",
    "ParsingReport",
    "LoadingReport",
]
