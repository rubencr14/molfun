"""Leakage-safe split of the KLIFS kinase chains for the pilot.

1. Extract every curated KLIFS chain to a single-chain mmCIF.
2. Cluster the SEQRES sequences with MMseqs2 (40% identity) and merge
   clusters that share a UniProt, so one protein is never on both sides.
3. Hold out a fraction of clusters (never the very large ones).
4. Drop any held-out chain with ≥ 40% identity over ≥ 80 residues to any
   training chain (dropped, never moved into training).
5. Pick evaluation targets: per held-out UniProt up to two structures, one
   per distinct DFG×αC state.

Run: ``python -m molfun.families.kinase.pilot_split`` (paths from ``.env``).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from molfun.data.leakage import leakage_hits, mmseqs_clusters, select_holdout

logger = logging.getLogger(__name__)

MIN_HELDOUT_UNIPROTS = 30
LEAK_RULE = {"max_identity": 0.4, "min_aln_len": 80}


@dataclass
class ClusterSplit:
    """Result of :func:`split_by_clusters`."""

    train_ids: list[str]
    heldout_ids: list[str]
    dropped_ids: list[str]
    report: dict = field(default_factory=dict)
    cluster_of: dict[str, str] = field(default_factory=dict)


@dataclass
class PilotSplit:
    """Result of :func:`build_pilot_split`."""

    train_ids: list[str]
    eval_targets: pd.DataFrame
    report: dict


def _merge_by_uniprot(clusters: dict[str, str], uniprot_of: dict[str, str]) -> dict[str, str]:
    """Union clusters that contain the same UniProt; returns id → merged cluster."""
    parent = {c: c for c in set(clusters.values())}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    by_uniprot = defaultdict(set)
    for sid, cluster in clusters.items():
        by_uniprot[uniprot_of[sid]].add(cluster)
    for group in by_uniprot.values():
        first, *rest = sorted(group)
        for other in rest:
            parent[find(other)] = find(first)
    return {sid: find(c) for sid, c in clusters.items()}


def split_by_clusters(
    seqs: dict[str, str],
    uniprot_of: dict[str, str],
    fraction: float,
    seed: int = 0,
    max_cluster_share: float = 0.02,
    min_seq_id: float = 0.4,
    coverage: float = 0.5,
) -> ClusterSplit:
    """Cluster hold-out with UniProt closure and the leakage drop rule."""
    merged = _merge_by_uniprot(mmseqs_clusters(seqs, min_seq_id, coverage), uniprot_of)
    held_clusters = select_holdout(merged, fraction, max_cluster_share, seed)
    held = sorted(s for s in seqs if merged[s] in held_clusters)
    train = sorted(s for s in seqs if merged[s] not in held_clusters)

    def subset(ids):
        return {s: seqs[s] for s in ids}

    hits = leakage_hits(subset(held), subset(train), **LEAK_RULE)
    dropped = sorted(set(hits["query"]))
    held = [s for s in held if s not in set(dropped)]
    remaining_hits = leakage_hits(subset(held), subset(train), **LEAK_RULE)
    report = {
        "n_structures": len(seqs),
        "n_clusters": len(set(merged.values())),
        "n_heldout_clusters": len(held_clusters),
        "n_train_structures": len(train),
        "n_heldout_structures": len(held),
        "n_dropped_for_leakage": len(dropped),
        "n_train_uniprots": len({uniprot_of[s] for s in train}),
        "n_heldout_uniprots": len({uniprot_of[s] for s in held}),
        "n_leak_hits_after_drop": len(remaining_hits),
        "fraction": fraction,
        "seed": seed,
        "max_cluster_share": max_cluster_share,
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        "leak_rule": LEAK_RULE,
    }
    return ClusterSplit(train, held, dropped, report, cluster_of=merged)


def write_split_files(
    out_dir: Path, split: ClusterSplit, targets: pd.DataFrame, groups: dict, report: dict
) -> None:
    """Write the split: ``train_ids.json``, ``heldout_ids.json`` (held-out
    chains kept after the leakage drop), ``clusters.json`` (id → merged
    cluster), ``uniprot_groups.json``, ``leakage_report.json``, ``eval_targets.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_ids.json").write_text(json.dumps(split.train_ids))
    (out_dir / "heldout_ids.json").write_text(json.dumps(split.heldout_ids))
    (out_dir / "clusters.json").write_text(json.dumps(split.cluster_of, sort_keys=True))
    (out_dir / "uniprot_groups.json").write_text(json.dumps(groups, indent=1, sort_keys=True))
    (out_dir / "leakage_report.json").write_text(json.dumps(report, indent=2, default=str))
    targets.to_csv(out_dir / "eval_targets.csv", index=False)


def select_eval_targets(rows: pd.DataFrame, per_uniprot: int = 2) -> pd.DataFrame:
    """Per UniProt, up to ``per_uniprot`` structures with distinct DFG×αC
    states (best resolution within a state); one best-resolution structure
    for proteins with no assigned state."""
    chosen = []
    for _, group in rows.groupby("uniprot", sort=True):
        ordered = group.sort_values(["resolution", "structure_id"])
        assigned = ordered[ordered["dfg"].notna() & ordered["ac_helix"].notna()]
        if assigned.empty:
            chosen.append(ordered.head(1))
        else:
            chosen.append(assigned.drop_duplicates(["dfg", "ac_helix"]).head(per_uniprot))
    return pd.concat(chosen, ignore_index=True)


def _extract_entry(job: tuple[str, list[tuple[str, str, str | None]], str]) -> list[tuple]:
    """Extract all requested chains of one entry (read the file once)."""
    import gemmi

    from molfun.families.kinase.chains import extract_chain_from_doc

    cif_gz, chains, out_dir = job
    results = []
    try:
        doc = gemmi.cif.read(cif_gz)
    except Exception as exc:  # noqa: BLE001 - reported per structure
        return [(sid, None, f"read: {exc}") for sid, _, _ in chains]
    for sid, chain, alt in chains:
        try:
            info = extract_chain_from_doc(doc, chain, Path(out_dir) / f"{sid}.cif", alt)
            results.append((sid, info, None))
        except Exception as exc:  # noqa: BLE001 - reported per structure
            results.append((sid, None, str(exc)))
    return results


def extract_all(
    table: pd.DataFrame, structures_dir: Path, chains_dir: Path, workers: int = 16
) -> tuple[dict, dict]:
    """Extract every table row in parallel; returns (id → ChainInfo, id → error)."""
    chains_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            str(structures_dir / f"{pdb}.cif.gz"),
            list(zip(g["structure_id"], g["chain"], g["alt"])),
            str(chains_dir),
        )
        for pdb, g in table.groupby("pdb")
    ]
    infos, errors = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for results in pool.map(_extract_entry, jobs, chunksize=8):
            for sid, info, error in results:
                if info is None:
                    errors[sid] = error
                else:
                    infos[sid] = info
    return infos, errors


def build_pilot_split(
    table: pd.DataFrame,
    structures_dir: Path,
    out_dir: Path,
    seed: int = 0,
    fraction: float = 0.15,
    min_observed: int = 100,
    workers: int = 16,
) -> PilotSplit:
    """Extract chains, split them leakage-safely and write the split files.

    Writes into ``out_dir``: ``chains/<id>.cif``, ``train_ids.json``,
    ``eval_targets.csv``, ``uniprot_groups.json``, ``leakage_report.json``,
    ``heldout_ids.json``, ``clusters.json`` (see :func:`write_split_files`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    chains_dir = out_dir / "chains"
    infos, errors = extract_all(table, structures_dir, chains_dir, workers)
    short = sorted(s for s, info in infos.items() if info.n_observed < min_observed)
    usable = table[table["structure_id"].isin(set(infos) - set(short))]
    seqs = {sid: infos[sid].seqres for sid in usable["structure_id"]}
    uniprot_of = dict(zip(usable["structure_id"], usable["uniprot"]))

    split = split_by_clusters(seqs, uniprot_of, fraction, seed)
    raised = False
    if split.report["n_heldout_uniprots"] < MIN_HELDOUT_UNIPROTS and fraction < 0.20:
        split = split_by_clusters(seqs, uniprot_of, 0.20, seed)
        raised = True

    held_rows = usable[usable["structure_id"].isin(split.heldout_ids)]
    targets = select_eval_targets(held_rows)
    targets = targets.assign(
        chain_cif=[str(chains_dir / f"{sid}.cif") for sid in targets["structure_id"]],
        seqres=[infos[sid].seqres for sid in targets["structure_id"]],
    )[
        [
            "structure_id", "uniprot", "kinase", "family", "group", "pdb", "chain_cif",
            "chain", "seqres", "pocket", "resolution", "dfg", "ac_helix",
        ]
    ]  # fmt: skip

    groups = defaultdict(list)
    for sid in split.train_ids:
        groups[uniprot_of[sid]].append(sid)

    report = {
        **split.report,
        "fraction_raised_to_0.20": raised,
        "n_table_rows": len(table),
        "n_extraction_failures": len(errors),
        "extraction_failures": dict(sorted(errors.items())[:50]),
        "min_observed": min_observed,
        "n_too_short": len(short),
        "n_eval_targets": len(targets),
        "n_eval_uniprots": targets["uniprot"].nunique(),
        "eval_states": targets.groupby(["dfg", "ac_helix"]).size().rename("n").reset_index().to_dict("records"),
        "dropped_for_leakage": split.dropped_ids,
    }  # fmt: skip
    write_split_files(out_dir, split, targets, groups, report)
    return PilotSplit(split.train_ids, targets, report)


def main(argv: list[str] | None = None) -> None:
    """CLI: build the pilot split from the paths in ``.env``."""
    from molfun.config.paths import pilot_paths
    from molfun.families.kinase.klifs import load_klifs_table

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    paths = pilot_paths()
    table = load_klifs_table(paths.kinase_data / "processed" / "master.csv")
    split = build_pilot_split(
        table,
        paths.kinase_data / "raw" / "structures" / "rcsb",
        paths.work_dir / "split",
        seed=args.seed,
        fraction=args.fraction,
        workers=args.workers,
    )
    print(
        json.dumps(
            {k: v for k, v in split.report.items() if not isinstance(v, (list, dict))}, indent=2
        )
    )


if __name__ == "__main__":
    main()
