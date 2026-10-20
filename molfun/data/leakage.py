"""Sequence-identity tools for leakage-safe splits (MMseqs2).

- :func:`mmseqs_clusters` groups sequences at a minimum identity.
- :func:`select_holdout` picks held-out clusters, never the very large ones.
- :func:`leakage_hits` lists held-out sequences too close to any training
  sequence, so they can be dropped from evaluation (never moved to training).

MMseqs2 must be installed; nothing here falls back to random splitting.
"""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

_HIT_COLUMNS = ["query", "target", "identity", "aln_len"]


def _require(mmseqs: str) -> str:
    path = shutil.which(mmseqs)
    if path is None:
        raise FileNotFoundError(f"MMseqs2 binary {mmseqs!r} not found; install mmseqs")
    return path


def _write_fasta(seqs: dict[str, str], path: Path) -> None:
    with path.open("w") as f:
        for name, seq in seqs.items():
            f.write(f">{name}\n{seq}\n")


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mmseqs failed ({' '.join(args[:2])}): {result.stderr[-2000:]}")


def mmseqs_clusters(
    seqs: dict[str, str],
    min_seq_id: float = 0.4,
    coverage: float = 0.5,
    mmseqs: str = "mmseqs",
    threads: int = 8,
) -> dict[str, str]:
    """Cluster sequences; returns id → representative id (its cluster id).

    Raises:
        FileNotFoundError: If MMseqs2 is not installed.
    """
    exe = _require(mmseqs)
    with tempfile.TemporaryDirectory(prefix="molfun-mmseqs-") as tmp:
        work = Path(tmp)
        _write_fasta(seqs, work / "in.fasta")
        _run(
            [
                exe, "easy-cluster", str(work / "in.fasta"), str(work / "res"), str(work / "tmp"),
                "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "0",
                "--threads", str(threads), "-v", "1",
            ]
        )  # fmt: skip
        pairs = pd.read_csv(work / "res_cluster.tsv", sep="\t", header=None, dtype=str)
    clusters = dict(zip(pairs[1], pairs[0]))
    return {name: clusters.get(name, name) for name in seqs}


def select_holdout(
    cluster_of: dict[str, str],
    fraction: float,
    max_cluster_share: float = 0.02,
    seed: int = 0,
) -> set[str]:
    """Pick ``round(fraction * n_eligible)`` held-out clusters, by cluster count.

    A cluster is eligible only if it holds at most ``max_cluster_share`` of
    all items, so big families (e.g. CDKs) stay in training.
    """
    sizes = Counter(cluster_of.values())
    total = len(cluster_of)
    eligible = sorted(c for c, n in sizes.items() if n / total <= max_cluster_share)
    k = round(fraction * len(eligible))
    return set(random.Random(seed).sample(eligible, k))


def leakage_hits(
    test: dict[str, str],
    train: dict[str, str],
    max_identity: float = 0.4,
    min_aln_len: int = 80,
    mmseqs: str = "mmseqs",
    threads: int = 8,
) -> pd.DataFrame:
    """Alignments from test to train with identity ≥ ``max_identity`` over
    ≥ ``min_aln_len`` residues (columns: query, target, identity, aln_len)."""
    if not test or not train:
        return pd.DataFrame(columns=_HIT_COLUMNS)
    exe = _require(mmseqs)
    with tempfile.TemporaryDirectory(prefix="molfun-mmseqs-") as tmp:
        work = Path(tmp)
        _write_fasta(test, work / "test.fasta")
        _write_fasta(train, work / "train.fasta")
        out = work / "hits.m8"
        _run(
            [
                exe, "easy-search", str(work / "test.fasta"), str(work / "train.fasta"), str(out),
                str(work / "tmp"), "--format-output", "query,target,fident,alnlen",
                "-s", "7.5", "-e", "10", "--max-seqs", "1000", "--threads", str(threads), "-v", "1",
            ]
        )  # fmt: skip
        if not out.exists() or out.stat().st_size == 0:
            return pd.DataFrame(columns=_HIT_COLUMNS)
        hits = pd.read_csv(out, sep="\t", header=None, names=_HIT_COLUMNS, dtype={0: str, 1: str})
    keep = (hits["identity"] >= max_identity) & (hits["aln_len"] >= min_aln_len)
    return hits[keep].reset_index(drop=True)
