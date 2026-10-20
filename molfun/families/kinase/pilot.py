"""Kinase pilot: train SimpleFold-100M on kinases, sample, compare with base.

    python -m molfun.families.kinase.pilot train
    python -m molfun.families.kinase.pilot evaluate --name base
    python -m molfun.families.kinase.pilot evaluate --name tuned --checkpoint <dir>
    python -m molfun.families.kinase.pilot report

Paths come from ``.env`` (see ``molfun.config.paths``); hyperparameters from
``recipes/kinase_pilot.yaml``. Every metric is computed on held-out kinase
families only, with predictions built from the sequence alone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

MIN_UNIPROTS = 30
_RECIPE = Path(__file__).resolve().parents[3] / "recipes" / "kinase_pilot.yaml"
_REPORT_METRICS = {
    "tm": "tm_mean",
    "lddt": "lddt_mean",
    "rmsd": "rmsd_mean",
    "state_top1": "state_top1_correct",
    "state_coverage@5": "state_recovered",
}


def load_pilot_config(path: str | Path = _RECIPE) -> dict:
    """Read the pilot recipe YAML."""
    import yaml

    return yaml.safe_load(Path(path).read_text())


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


def _state_name(dfg, ac) -> str | None:
    return f"{dfg}/{ac}" if isinstance(dfg, str) and isinstance(ac, str) else None


def evaluate_target(
    model,
    target: pd.Series,
    labeler,
    pocket_dir: Path,
    num_samples: int,
    num_steps: int,
    tau: float,
    lddt_threshold: float,
    seed: int = 0,
) -> dict:
    """Sample one held-out target from its sequence and score every sample.

    Returns accuracy (mean and best over samples) and states: the label of
    each sample, whether the first sample has the experimental state
    (``state_top1_correct``) and whether any sample has it with lDDT-Cα ≥
    ``lddt_threshold`` (``state_recovered``). State fields are None when the
    target has no experimental state or its motif cannot be located.
    """
    import gemmi

    from molfun.eval.structure import ca_rmsd, lddt_ca, paired_ca, tm_score
    from molfun.families.kinase.states import (
        locate_motif_by_numbering,
        motif_features,
        read_pocket_residues,
    )

    start = time.time()
    samples = model.sample(
        target["seqres"], num_samples=num_samples, num_steps=num_steps, tau=tau, seed=seed
    )
    elapsed = time.time() - start

    reference = gemmi.read_structure(target["chain_cif"])
    mol2 = pocket_dir / f"{target['structure_id']}.mol2"
    idx = (
        locate_motif_by_numbering(target["pocket"], read_pocket_residues(mol2), reference)
        if mol2.is_file()
        else None
    )
    truth = _state_name(target.get("dfg"), target.get("ac_helix"))

    tms, lddts, rmsds, states, n_pairs = [], [], [], [], 0
    for sample in samples:
        p, r = paired_ca(sample, target["chain_cif"], target["chain"])
        n_pairs = len(r)
        tms.append(tm_score(p, r))
        lddts.append(lddt_ca(p, r))
        rmsds.append(ca_rmsd(p, r))
        if idx is not None:
            block = gemmi.cif.read_string(sample.mmcif).sole_block()
            feats = pd.DataFrame([motif_features(gemmi.make_structure_from_block(block), idx)])
            label = labeler.predict(feats).iloc[0]
            states.append(_state_name(label["dfg"], label["ac_helix"]))

    scored = truth is not None and idx is not None
    return {
        "structure_id": target["structure_id"],
        "uniprot": target["uniprot"],
        "state": truth,
        "n_pairs": n_pairs,
        "tm_mean": float(np.mean(tms)),
        "tm_best": float(np.max(tms)),
        "lddt_mean": float(np.mean(lddts)),
        "lddt_best": float(np.max(lddts)),
        "rmsd_mean": float(np.mean(rmsds)),
        "rmsd_best": float(np.min(rmsds)),
        "sampled_states": states,
        "state_top1_correct": (states[0] == truth) if scored else None,
        "state_recovered": (
            any(s == truth and lddt >= lddt_threshold for s, lddt in zip(states, lddts))
            if scored
            else None
        ),
        "sampling_seconds": round(elapsed, 2),
    }


def _load_model(name: str, checkpoint: Path | None):
    from molfun.models.structure import MolfunStructureModel

    model = MolfunStructureModel.from_pretrained(name, device="cuda", head="generative")
    if checkpoint is not None:
        model.load(str(checkpoint))
    return model


def evaluate_pilot(cfg: dict, checkpoint: Path | None, name: str) -> pd.DataFrame:
    """Score every eval target; writes ``<work>/eval/<name>.csv``."""
    from molfun.config.paths import pilot_paths
    from molfun.families.kinase.states import StateLabeler

    paths = pilot_paths()
    targets = pd.read_csv(
        paths.work_dir / "split" / "eval_targets.csv", dtype={"structure_id": str}
    )
    targets = targets.astype(object).where(targets.notna(), None)
    labeler = StateLabeler.load(paths.work_dir / "state_labeler.json")
    pocket_dir = paths.kinase_data / "raw" / "structures" / "klifs" / "pocket"
    model = _load_model(cfg["model"], checkpoint)

    rows = [
        evaluate_target(
            model,
            target,
            labeler,
            pocket_dir,
            num_samples=cfg["sample"]["num_samples"],
            num_steps=cfg["sample"]["num_steps"],
            tau=cfg["sample"]["tau"],
            lddt_threshold=cfg["eval"]["state_lddt_threshold"],
            seed=int(target["structure_id"]),
        )
        for _, target in targets.iterrows()
    ]
    out = pd.DataFrame(rows)
    (paths.work_dir / "eval").mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.work_dir / "eval" / f"{name}.csv", index=False)
    return out


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------


def _per_uniprot(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[["uniprot", column]].dropna()
    return values.groupby("uniprot")[column].mean().astype(float)


def pilot_report(
    base: pd.DataFrame, tuned: pd.DataFrame, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Paired comparison on the same targets, averaged per UniProt first; 95%
    CIs of the mean difference from ``n_boot`` bootstrap resamples of UniProts.

    Raises:
        ValueError: If the two frames do not cover the same targets.
    """
    if set(base["structure_id"]) != set(tuned["structure_id"]):
        raise ValueError("base and tuned must be scored on the same targets")
    rng = np.random.default_rng(seed)
    report = {
        "n_targets": len(base),
        "n_uniprots": int(base["uniprot"].nunique()),
        "underpowered": bool(base["uniprot"].nunique() < MIN_UNIPROTS),
    }
    for key, column in _REPORT_METRICS.items():
        b, t = _per_uniprot(base, column), _per_uniprot(tuned, column)
        common = b.index.intersection(t.index)
        diff = (t[common] - b[common]).to_numpy()
        if len(diff) == 0:
            report[key] = {"base": None, "tuned": None, "delta": None, "ci95": None}
            continue
        boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
        report[key] = {
            "base": float(b[common].mean()),
            "tuned": float(t[common].mean()),
            "delta": float(diff.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        }
    accuracy_held = all(
        report[k]["delta"] is not None and report[k]["delta"] >= 0 for k in ("tm", "lddt")
    )
    states_better = any(
        report[k]["delta"] is not None and report[k]["delta"] > 0
        for k in ("state_top1", "state_coverage@5")
    )
    report["go"] = bool(accuracy_held and states_better)
    return report


def write_report(report: dict, out_dir: Path) -> None:
    """Write ``pilot_report.json`` and a readable ``pilot_report.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pilot_report.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# Kinase pilot — SimpleFold-100M base vs kinase fine-tune",
        "",
        "100M trainable trunk + frozen ESM2-3B. Held-out kinase families only; "
        "predictions from sequence alone; means per UniProt, 95% CI by bootstrap over UniProts.",
        "",
        f"Targets: {report['n_targets']} · UniProts: {report['n_uniprots']}"
        + (" · **underpowered (< 30 proteins)**" if report["underpowered"] else ""),
        "",
        "| Metric | Base | Tuned | Δ | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for key in _REPORT_METRICS:
        m = report[key]
        if m["delta"] is None:
            lines.append(f"| {key} | – | – | – | – |")
            continue
        lines.append(
            f"| {key} | {m['base']:.3f} | {m['tuned']:.3f} | {m['delta']:+.3f} | "
            f"[{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}] |"
        )
    lines += [
        "",
        f"**Go/no-go (spec §5.8): {'GO' if report['go'] else 'NO-GO'}** — go needs Δ ≥ 0 on "
        "TM and lDDT and Δ > 0 on a state metric.",
    ]
    (out_dir / "pilot_report.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def train_pilot(cfg: dict) -> Path:
    """Fine-tune on the training UniProts; returns the saved checkpoint dir
    (EMA weights applied before saving)."""
    import torch
    from torch.utils.data import DataLoader

    from molfun.backends.simplefold.records import SimpleFoldRecordDataset, collate_records
    from molfun.config.paths import pilot_paths
    from molfun.training import FullFinetune

    paths = pilot_paths()
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    groups = json.loads((paths.work_dir / "split" / "uniprot_groups.json").read_text())
    dataset = SimpleFoldRecordDataset(
        paths.work_dir / "records" / "tokens",
        paths.work_dir / "records" / "targets",
        groups,
        max_tokens=t["max_tokens"],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=t["num_workers"],
        collate_fn=collate_records,
        persistent_workers=t["num_workers"] > 0,
    )
    model = _load_model(cfg["model"], None)
    strategy = FullFinetune(
        lr=t["lr"],
        layer_lr_decay=t["layer_lr_decay"],
        warmup_steps=t["warmup_steps"],
        scheduler=t["scheduler"],
        ema_decay=t["ema_decay"],
        grad_clip=t["grad_clip"],
        amp_dtype=t["amp_dtype"],
    )
    run_dir = paths.work_dir / "run"
    history = model.fit(loader, strategy=strategy, epochs=t["epochs"])
    strategy.apply_ema(model)
    final = run_dir / "final"
    model.save(str(final))
    (run_dir / "history.json").write_text(json.dumps(history, indent=1, default=float))
    return final


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (see module docstring)."""
    parser = argparse.ArgumentParser(description="Kinase pilot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--name", required=True)
    ev.add_argument("--checkpoint", type=Path, default=None)
    sub.add_parser("report")
    args = parser.parse_args(argv)
    cfg = load_pilot_config()

    if args.cmd == "train":
        print(train_pilot(cfg))
    elif args.cmd == "evaluate":
        frame = evaluate_pilot(cfg, args.checkpoint, args.name)
        print(frame[["structure_id", "tm_mean", "lddt_mean", "state_top1_correct"]].to_string())
    else:
        from molfun.config.paths import pilot_paths

        work = pilot_paths().work_dir
        base = pd.read_csv(work / "eval" / "base.csv", dtype={"structure_id": str})
        tuned = pd.read_csv(work / "eval" / "tuned.csv", dtype={"structure_id": str})
        report = pilot_report(base, tuned, n_boot=cfg["eval"]["bootstrap"])
        write_report(report, work)
        print((work / "pilot_report.md").read_text())


if __name__ == "__main__":
    main()
