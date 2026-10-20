"""Fit the DFG/αC state labeler on pilot training chains, validate on held-out ones.

Training chains are labelled with the published KLIFS states; the labeler is
then scored on held-out chains (kinase families it never saw). The pilot
uses predicted-state metrics only if the labeler passes its gates.

Run: ``python -m molfun.families.kinase.fit_labeler`` (paths from ``.env``).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

GATES = {"dfg": 0.90, "ac_helix": 0.80}


def chain_seqres(structure) -> str:
    """SEQRES one-letter sequence of the first chain of a structure."""
    from molfun.families.kinase.chains import seqres_one_letter

    structure.setup_entities()
    polymer = structure[0][0].get_polymer()
    entity = structure.get_entity_of(polymer)
    names = list(entity.full_sequence) if entity and entity.full_sequence else []
    return seqres_one_letter(names or [r.name for r in polymer])


def _features_one(job: tuple[str, str, str, str]) -> dict | None:
    import gemmi

    from molfun.families.kinase.states import (
        locate_motif_by_numbering,
        motif_features,
        read_pocket_residues,
    )

    structure_id, cif, pocket, mol2 = job
    if not (Path(cif).is_file() and Path(mol2).is_file()):
        return None
    structure = gemmi.read_structure(cif)
    idx = locate_motif_by_numbering(pocket, read_pocket_residues(mol2), structure)
    if idx is None:
        return None
    return {"structure_id": structure_id, **motif_features(structure, idx)}


def compute_features(
    rows: pd.DataFrame, chains_dir: Path, pocket_dir: Path, workers: int = 16
) -> pd.DataFrame:
    """Motif features for each row (``structure_id``, ``pocket``), with the
    motif located through the KLIFS pocket mol2 numbering
    (``pocket_dir/<id>.mol2``); rows whose motif is not found are left out."""
    jobs = [
        (sid, str(chains_dir / f"{sid}.cif"), pocket, str(pocket_dir / f"{sid}.mol2"))
        for sid, pocket in zip(rows["structure_id"], rows["pocket"])
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        found = [r for r in pool.map(_features_one, jobs, chunksize=32) if r is not None]
    return pd.DataFrame(found, columns=None if found else ["structure_id"])


def fit_and_validate(
    table: pd.DataFrame, split_dir: Path, pocket_dir: Path, out_path: Path, workers: int = 16
) -> dict:
    """Fit on training chains, score on held-out chains, save labeler + report."""
    from sklearn.metrics import balanced_accuracy_score

    from molfun.families.kinase.states import StateLabeler

    chains_dir = split_dir / "chains"
    train_ids = set(json.loads((split_dir / "train_ids.json").read_text()))
    dropped = set(
        json.loads((split_dir / "leakage_report.json").read_text())["dropped_for_leakage"]
    )
    extracted = {p.stem for p in chains_dir.glob("*.cif")}
    held_ids = extracted - train_ids - dropped

    labels = table.set_index("structure_id")[["dfg", "ac_helix"]]
    train_rows = table[table["structure_id"].isin(train_ids)]
    held_rows = table[table["structure_id"].isin(held_ids)]
    train = compute_features(train_rows, chains_dir, pocket_dir, workers)
    held = compute_features(held_rows, chains_dir, pocket_dir, workers)
    train = train.join(labels, on="structure_id")
    held = held.join(labels, on="structure_id")

    labeler = StateLabeler().fit(train, train["dfg"], train["ac_helix"])
    predicted = labeler.predict(held)
    report = {"n_train": len(train), "n_heldout": len(held), "gates": GATES}
    for name in ("dfg", "ac_helix"):
        known = held[name].notna()
        score = balanced_accuracy_score(held.loc[known, name], predicted.loc[known, name])
        report[f"{name}_balanced_accuracy"] = round(float(score), 4)
        report[f"{name}_n_scored"] = int(known.sum())
        report[f"{name}_pass"] = bool(score >= GATES[name])
    report["n_train_motif_not_found"] = len(train_ids) - len(train)
    report["n_heldout_motif_not_found"] = len(held_ids) - len(held)

    labeler.save(out_path)
    out_path.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI: fit and validate the labeler on the pilot split."""
    from molfun.config.paths import pilot_paths
    from molfun.families.kinase.klifs import load_klifs_table

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    paths = pilot_paths()
    table = load_klifs_table(paths.kinase_data / "processed" / "master.csv")
    report = fit_and_validate(
        table,
        paths.work_dir / "split",
        paths.kinase_data / "raw" / "structures" / "klifs" / "pocket",
        paths.work_dir / "state_labeler.json",
        args.workers,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
