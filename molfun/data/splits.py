"""
Data splitting strategies for protein datasets.

Provides splits that avoid data leakage caused by sequence homology,
which is critical for meaningful generalization benchmarks.
"""

from __future__ import annotations
from typing import Optional
import random
import subprocess
import tempfile
from pathlib import Path

from torch.utils.data import Dataset, Subset


class DataSplitter:
    """
    Static methods for splitting protein datasets.

    All methods return (train, val, test) Subsets of the input dataset.

    Usage:
        train, val, test = DataSplitter.random(dataset)
        train, val, test = DataSplitter.by_sequence_identity(dataset, threshold=0.3)
        train, val, test = DataSplitter.by_family(dataset, families)
    """

    @staticmethod
    def random(
        dataset: Dataset,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
    ) -> tuple[Subset, Subset, Subset]:
        """Simple random split. Fast but ignores sequence homology."""
        n = len(dataset)
        indices = list(range(n))
        rng = random.Random(seed)
        rng.shuffle(indices)

        n_test = int(n * test_frac)
        n_val = int(n * val_frac)

        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

        return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

    @staticmethod
    def by_sequence_identity(
        dataset,
        threshold: float = 0.3,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        mmseqs_path: str = "mmseqs",
    ) -> tuple[Subset, Subset, Subset]:
        """
        Split by sequence clustering so that no two sequences in different
        splits share more than `threshold` identity.

        Requires MMseqs2 installed (https://github.com/soedinglab/MMseqs2).
        Falls back to random split if MMseqs2 is not available.

        Args:
            dataset: Must expose a `.sequences` property returning list[str].
            threshold: Sequence identity threshold (0.0–1.0). Clusters above
                       this threshold are kept together.
            val_frac: Fraction of clusters for validation.
            test_frac: Fraction of clusters for test.
            seed: Random seed for cluster assignment.
            mmseqs_path: Path to mmseqs binary.
        """
        sequences = dataset.sequences
        if not sequences:
            raise ValueError("Dataset has no sequences for identity-based splitting.")

        try:
            clusters = _cluster_mmseqs(sequences, threshold, mmseqs_path)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return DataSplitter.random(dataset, val_frac, test_frac, seed)

        return _split_by_clusters(dataset, clusters, val_frac, test_frac, seed)

    @staticmethod
    def by_family(
        dataset,
        families: list[str],
        val_families: Optional[set[str]] = None,
        test_families: Optional[set[str]] = None,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
    ) -> tuple[Subset, Subset, Subset]:
        """
        Split by protein family: entire families go to one split.

        Args:
            dataset: The dataset to split.
            families: list[str] of length len(dataset), family label per entry.
            val_families: Explicit set of families for validation. If None,
                         families are randomly assigned.
            test_families: Explicit set of families for test. If None,
                          families are randomly assigned.
            val_frac: Fraction of families for validation (if auto-assigning).
            test_frac: Fraction of families for test (if auto-assigning).
            seed: Random seed.
        """
        unique_families = sorted(set(families))
        rng = random.Random(seed)

        if val_families is None or test_families is None:
            rng.shuffle(unique_families)
            n_test = max(1, int(len(unique_families) * test_frac))
            n_val = max(1, int(len(unique_families) * val_frac))
            test_families = set(unique_families[:n_test])
            val_families = set(unique_families[n_test:n_test + n_val])

        train_idx, val_idx, test_idx = [], [], []
        for i, fam in enumerate(families):
            if fam in test_families:
                test_idx.append(i)
            elif fam in val_families:
                val_idx.append(i)
            else:
                train_idx.append(i)

        return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

    @staticmethod
    def temporal(
        dataset,
        years: list[int],
        val_cutoff: int = 2019,
        test_cutoff: int = 2020,
    ) -> tuple[Subset, Subset, Subset]:
        """
        Temporal split based on deposition year.

        Train: year < val_cutoff
        Val:   val_cutoff <= year < test_cutoff
        Test:  year >= test_cutoff

        Args:
            dataset: The dataset to split.
            years: list[int] of length len(dataset), year per entry.
            val_cutoff: Year boundary for validation.
            test_cutoff: Year boundary for test.
        """
        train_idx, val_idx, test_idx = [], [], []
        for i, y in enumerate(years):
            if y >= test_cutoff:
                test_idx.append(i)
            elif y >= val_cutoff:
                val_idx.append(i)
            else:
                train_idx.append(i)

        return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


# ------------------------------------------------------------------
# Internal: MMseqs2 clustering
# ------------------------------------------------------------------

def _cluster_mmseqs(
    sequences: list[str],
    threshold: float,
    mmseqs_path: str,
) -> list[int]:
    """
    Cluster sequences with MMseqs2.

    Returns list[int] of cluster IDs (same length as sequences).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fasta = tmpdir / "seqs.fasta"
        db = tmpdir / "seqdb"
        clu = tmpdir / "clu"
        tsv = tmpdir / "clu.tsv"

        with open(fasta, "w") as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq_{i}\n{seq}\n")

        subprocess.run(
            [mmseqs_path, "createdb", str(fasta), str(db)],
            check=True, capture_output=True,
        )
        subprocess.run(
            [mmseqs_path, "cluster", str(db), str(clu), str(tmpdir),
             "--min-seq-id", str(threshold), "-c", "0.8", "--cov-mode", "0"],
            check=True, capture_output=True,
        )
        subprocess.run(
            [mmseqs_path, "createtsv", str(db), str(db), str(clu), str(tsv)],
            check=True, capture_output=True,
        )

        # Parse TSV: representative \t member
        rep_to_cluster = {}
        cluster_id = 0
        member_to_cluster = {}

        with open(tsv) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                rep, member = parts[0], parts[1]
                if rep not in rep_to_cluster:
                    rep_to_cluster[rep] = cluster_id
                    cluster_id += 1
                member_to_cluster[member] = rep_to_cluster[rep]

    clusters = []
    for i in range(len(sequences)):
        key = f"seq_{i}"
        clusters.append(member_to_cluster.get(key, i))

    return clusters


def _split_by_clusters(
    dataset,
    clusters: list[int],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[Subset, Subset, Subset]:
    """Assign entire clusters to train/val/test."""
    unique = sorted(set(clusters))
    rng = random.Random(seed)
    rng.shuffle(unique)

    n_test = max(1, int(len(unique) * test_frac))
    n_val = max(1, int(len(unique) * val_frac))

    test_clusters = set(unique[:n_test])
    val_clusters = set(unique[n_test:n_test + n_val])

    train_idx, val_idx, test_idx = [], [], []
    for i, c in enumerate(clusters):
        if c in test_clusters:
            test_idx.append(i)
        elif c in val_clusters:
            val_idx.append(i)
        else:
            train_idx.append(i)

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
