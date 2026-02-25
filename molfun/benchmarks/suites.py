"""
Declarative benchmark suite definitions.

A ``BenchmarkTask`` describes *what* to evaluate (dataset + metrics + split).
A ``BenchmarkSuite`` groups multiple tasks into a reproducible evaluation
protocol.  Pre-built suites are provided for common datasets; users can
also build custom ones.

Follows Open/Closed Principle: new suites are added without modifying
existing code — just call ``BenchmarkSuite.custom()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    STRUCTURE = "structure"
    DOCKING = "docking"


@dataclass(frozen=True)
class BenchmarkTask:
    """
    Immutable definition of a single evaluation task.

    Attributes:
        name: Human-readable identifier (e.g. ``pdbbind_core_v2020``).
        data_source: Path or URI to the evaluation data.
        split: Split strategy identifier or path to a split file.
        metrics: Metric names (keys of ``METRIC_REGISTRY``).
        task_type: Determines how predictions are interpreted.
        max_samples: Cap for quick smoke-test runs (None = all).
        description: Optional one-liner for reports.
    """

    name: str
    data_source: str
    split: str = "test"
    metrics: tuple[str, ...] = ("mae", "pearson")
    task_type: TaskType = TaskType.REGRESSION
    max_samples: Optional[int] = None
    description: str = ""


@dataclass
class BenchmarkSuite:
    """
    Ordered collection of ``BenchmarkTask`` instances.

    Pre-built class methods provide standard community benchmarks;
    ``custom()`` lets users define their own.
    """

    name: str
    tasks: list[BenchmarkTask] = field(default_factory=list)
    description: str = ""

    # ------------------------------------------------------------------
    # Factory methods for standard benchmarks
    # ------------------------------------------------------------------

    @classmethod
    def pdbbind(
        cls,
        index_path: str = "data/pdbbind/INDEX_refined_data.2020",
        structures_dir: str = "data/pdbbind/structures",
        split: str = "identity_30",
    ) -> BenchmarkSuite:
        """
        PDBbind benchmark — protein-ligand binding affinity prediction.

        Standard in drug-discovery ML: predict pKd/pKi from structure.
        """
        return cls(
            name="PDBbind-v2020",
            description="Protein-ligand binding affinity (pKd/pKi regression)",
            tasks=[
                BenchmarkTask(
                    name="pdbbind_core_v2020",
                    data_source=index_path,
                    split=split,
                    metrics=("mae", "rmse", "pearson", "spearman"),
                    task_type=TaskType.REGRESSION,
                    description="PDBbind core set (285 complexes)",
                ),
                BenchmarkTask(
                    name="pdbbind_refined_v2020",
                    data_source=index_path,
                    split=split,
                    metrics=("mae", "rmse", "pearson", "spearman", "r2"),
                    task_type=TaskType.REGRESSION,
                    description="PDBbind refined set (4,852 complexes)",
                ),
            ],
        )

    @classmethod
    def atom3d_lba(
        cls,
        data_dir: str = "data/atom3d/LBA",
        split: str = "sequence_30",
    ) -> BenchmarkSuite:
        """
        ATOM3D Ligand Binding Affinity task.

        Townsend et al. (2021) — 3D molecular learning benchmark.
        """
        return cls(
            name="ATOM3D-LBA",
            description="ATOM3D Ligand Binding Affinity (regression)",
            tasks=[
                BenchmarkTask(
                    name="atom3d_lba_identity30",
                    data_source=data_dir,
                    split=split,
                    metrics=("mae", "rmse", "pearson", "spearman"),
                    task_type=TaskType.REGRESSION,
                    description="ATOM3D LBA, 30% sequence identity split",
                ),
            ],
        )

    @classmethod
    def atom3d_psr(
        cls,
        data_dir: str = "data/atom3d/PSR",
        split: str = "year",
    ) -> BenchmarkSuite:
        """ATOM3D Protein Structure Ranking task."""
        return cls(
            name="ATOM3D-PSR",
            description="ATOM3D Protein Structure Ranking",
            tasks=[
                BenchmarkTask(
                    name="atom3d_psr",
                    data_source=data_dir,
                    split=split,
                    metrics=("spearman", "pearson"),
                    task_type=TaskType.REGRESSION,
                    description="ATOM3D PSR, temporal split",
                ),
            ],
        )

    @classmethod
    def flip(
        cls,
        data_dir: str = "data/flip",
    ) -> BenchmarkSuite:
        """
        FLIP: Fitness Landscape Inference for Proteins (Dallago et al. 2021).

        Multiple protein fitness prediction tasks with controlled splits.
        """
        landscapes = [
            ("aav", "AAV capsid fitness"),
            ("gb1", "GB1 binding fitness"),
            ("meltome", "Meltome thermostability"),
        ]
        tasks = [
            BenchmarkTask(
                name=f"flip_{lname}",
                data_source=f"{data_dir}/{lname}",
                split="sampled",
                metrics=("spearman", "mae"),
                task_type=TaskType.REGRESSION,
                description=desc,
            )
            for lname, desc in landscapes
        ]
        return cls(name="FLIP", description="Protein fitness landscape prediction", tasks=tasks)

    @classmethod
    def structure_quality(
        cls,
        targets_dir: str = "data/casp15/targets",
    ) -> BenchmarkSuite:
        """
        Structure prediction quality benchmark (CASP-style metrics).

        Evaluates predicted 3D coordinates against experimental structures.
        """
        return cls(
            name="StructureQuality",
            description="Structure prediction accuracy (CASP-style)",
            tasks=[
                BenchmarkTask(
                    name="structure_gdt",
                    data_source=targets_dir,
                    split="all",
                    metrics=("gdt_ts", "tm_score", "lddt", "coord_rmsd"),
                    task_type=TaskType.STRUCTURE,
                    description="GDT-TS, TM-score, lDDT, RMSD on target structures",
                ),
            ],
        )

    @classmethod
    def custom(
        cls,
        name: str,
        tasks: list[BenchmarkTask],
        description: str = "",
    ) -> BenchmarkSuite:
        """Build a suite from user-defined tasks."""
        return cls(name=name, tasks=tasks, description=description)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def task_names(self) -> list[str]:
        return [t.name for t in self.tasks]

    def summary(self) -> str:
        lines = [f"Suite: {self.name} — {self.description}", f"Tasks: {len(self.tasks)}"]
        for t in self.tasks:
            lines.append(f"  - {t.name}: {', '.join(t.metrics)} ({t.task_type.value})")
        return "\n".join(lines)
