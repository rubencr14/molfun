"""Leakage-safe pilot split: cluster hold-out, UniProt closure, eval targets."""

import random
import shutil

import pandas as pd
import pytest

from molfun.data.leakage import leakage_hits
from molfun.families.kinase.pilot_split import select_eval_targets, split_by_clusters

AA = "ACDEFGHIKLMNPQRSTVWY"
needs_mmseqs = pytest.mark.skipif(shutil.which("mmseqs") is None, reason="needs mmseqs")


def _family_sequences(seed=0, n_families=6, members=3, length=150):
    rng = random.Random(seed)
    seqs, uniprot_of = {}, {}
    for f in range(n_families):
        base = "".join(rng.choice(AA) for _ in range(length))
        for m in range(members):
            seq = list(base)
            for i in rng.sample(range(length), length // 10):
                seq[i] = rng.choice(AA)
            sid = f"f{f}m{m}"
            seqs[sid] = "".join(seq)
            uniprot_of[sid] = f"P{f}_{m}"
    # one protein crystallised twice, as two constructs of different length
    seqs["f0m0_long"] = seqs["f0m0"] + "".join(rng.choice(AA) for _ in range(120))
    uniprot_of["f0m0_long"] = "P0_0"
    return seqs, uniprot_of


@needs_mmseqs
def test_pilot_split_no_leak():
    seqs, uniprot_of = _family_sequences()

    split = split_by_clusters(seqs, uniprot_of, fraction=0.34, seed=0, max_cluster_share=1.0)

    train, held = set(split.train_ids), set(split.heldout_ids)
    assert held, "nothing held out"
    assert train | held | set(split.dropped_ids) == set(seqs)
    assert not {uniprot_of[s] for s in train} & {uniprot_of[s] for s in held}
    assert leakage_hits({s: seqs[s] for s in held}, {s: seqs[s] for s in train}).empty
    assert split.report["n_heldout_structures"] == len(held)


@needs_mmseqs
def test_pilot_split_keeps_constructs_of_one_protein_together():
    seqs, uniprot_of = _family_sequences()

    split = split_by_clusters(seqs, uniprot_of, fraction=0.5, seed=1, max_cluster_share=1.0)

    sides = {
        "train" if s in split.train_ids else "held" if s in split.heldout_ids else "dropped"
        for s in ("f0m0", "f0m0_long")
    }
    assert len(sides) == 1


def test_select_eval_targets_one_per_state():
    rows = pd.DataFrame(
        {
            "structure_id": ["1", "2", "3", "4", "5", "6"],
            "uniprot": ["P1", "P1", "P1", "P2", "P2", "P3"],
            "dfg": ["in", "in", "out", None, "in", None],
            "ac_helix": ["in", "in", "in", None, "out", None],
            "resolution": [2.0, 1.5, 2.2, 1.0, 3.0, 2.0],
        }
    )

    targets = select_eval_targets(rows, per_uniprot=2)

    assert set(targets["structure_id"]) == {"2", "3", "5", "6"}
