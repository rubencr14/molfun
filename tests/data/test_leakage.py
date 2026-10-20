"""Sequence clustering, hold-out selection and the leakage check (MMseqs2)."""

import random
import shutil

import pytest

from molfun.data.leakage import leakage_hits, mmseqs_clusters, select_holdout

needs_mmseqs = pytest.mark.skipif(shutil.which("mmseqs") is None, reason="needs mmseqs")
AA = "ACDEFGHIKLMNPQRSTVWY"


def _random_seq(rng, n):
    return "".join(rng.choice(AA) for _ in range(n))


def _mutate(rng, seq, identity):
    """Copy of seq with exactly round((1 - identity) * len) substitutions."""
    out = list(seq)
    for i in rng.sample(range(len(seq)), round((1 - identity) * len(seq))):
        out[i] = rng.choice([a for a in AA if a != seq[i]])
    return "".join(out)


def test_mmseqs_missing_raises():
    with pytest.raises(FileNotFoundError, match="mmseqs"):
        mmseqs_clusters({"a": "ACDEFGHIKL"}, mmseqs="mmseqs-not-installed")


@needs_mmseqs
def test_mmseqs_clusters_groups_close_sequences():
    rng = random.Random(0)
    base = _random_seq(rng, 150)
    seqs = {"a": base, "b": _mutate(rng, base, 0.9), "c": _random_seq(rng, 150)}

    clusters = mmseqs_clusters(seqs)

    assert clusters["a"] == clusters["b"]
    assert clusters["c"] != clusters["a"]


@needs_mmseqs
def test_leakage_hits_flags_close_homolog():
    rng = random.Random(1)
    base = _random_seq(rng, 120)
    train = {"t1": base}
    test = {"close": _mutate(rng, base, 0.6), "far": _mutate(rng, base, 0.3)}

    hits = leakage_hits(test, train)

    assert set(hits["query"]) == {"close"}
    assert (hits["identity"] >= 0.4).all()
    assert (hits["aln_len"] >= 80).all()


@needs_mmseqs
def test_leakage_hits_finds_homolog_crowded_out_by_decoys():
    """1,100 decoys share a 60-residue exact segment with the query, so they
    outrank the real homolog (55% over 120 residues) in a prefilter; the
    homolog must still be found — a candidate cap must not hide it."""
    rng = random.Random(3)
    query = _random_seq(rng, 120)
    train = {f"decoy{i}": query[:60] + _random_seq(rng, 90) for i in range(1100)}
    train["homolog"] = _mutate(rng, query, 0.55)

    hits = leakage_hits({"q": query}, train)

    assert "homolog" in set(hits["target"])


@needs_mmseqs
def test_leakage_hits_ignores_short_alignments():
    rng = random.Random(2)
    shared = _random_seq(rng, 40)
    train = {"t1": shared + _random_seq(rng, 150)}
    test = {"q": _random_seq(rng, 150) + shared}

    assert leakage_hits(test, train).empty


def test_select_holdout_respects_share_cap():
    cluster_of = {f"big{i}": "BIG" for i in range(50)}
    cluster_of.update({f"s{i}": f"C{i}" for i in range(50)})

    held = select_holdout(cluster_of, fraction=0.5, max_cluster_share=0.02, seed=0)

    assert "BIG" not in held
    assert len(held) == 25  # half of the 50 eligible small clusters


def test_select_holdout_is_seeded():
    cluster_of = {f"s{i}": f"C{i}" for i in range(40)}

    a = select_holdout(cluster_of, fraction=0.25, seed=3)
    b = select_holdout(cluster_of, fraction=0.25, seed=3)

    assert a == b
