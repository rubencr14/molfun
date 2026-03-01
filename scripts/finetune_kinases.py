#!/usr/bin/env python
"""
End-to-end fine-tuning of OpenFold on human kinases.

All parameters are read from a YAML config file. CLI flags can override
any value for quick experiments.

Usage:
    python scripts/finetune_kinases.py
    python scripts/finetune_kinases.py --config configs/custom.yaml
    python scripts/finetune_kinases.py --epochs 5 --rank 4
    python scripts/finetune_kinases.py --minio --max-structures 50
"""

from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
logging.getLogger("torch.fx._symbolic_trace").setLevel(logging.ERROR)

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from molfun.models import MolfunStructureModel
from molfun.backends.openfold import OpenFoldFeaturizer
from molfun.data.collections import fetch_collection, search_collection
from molfun.data.datasets import StructureDataset
from molfun.data.sources.pdb import PDBFetcher
from molfun.data.splits import DataSplitter
from molfun.training import LoRAFinetune, HeadOnlyFinetune, PartialFinetune, FullFinetune
from molfun.storage import MinioStorage
from molfun.tracking import ExperimentRegistry


DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "finetune_kinases.yaml"

STRATEGIES = {
    "lora": LoRAFinetune,
    "head_only": HeadOnlyFinetune,
    "partial": PartialFinetune,
    "full": FullFinetune,
}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    overrides = {
        ("data", "max_structures"): args.max_structures,
        ("data", "max_seq_len"): args.max_seq_len,
        ("data", "cache_dir"): args.data_dir,
        ("training", "epochs"): args.epochs,
        ("training", "rank"): args.rank,
        ("training", "lr_lora"): args.lr_lora,
        ("training", "lr_head"): args.lr_head,
        ("training", "grad_accum"): args.grad_accum,
        ("model", "device"): args.device,
        ("output", "checkpoint_dir"): args.output_dir,
        ("output", "push_to_hub"): args.push_to_hub,
    }
    for (section, key), val in overrides.items():
        if val is not None:
            cfg.setdefault(section, {})[key] = val
    if args.minio:
        cfg.setdefault("storage", {})["minio"] = True
    if args.no_registry:
        cfg.setdefault("storage", {})["registry"] = False
    if args.resume_from:
        cfg.setdefault("training", {})["resume_from"] = args.resume_from
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune OpenFold on protein collections")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--max-structures", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--lr-lora", type=float, default=None)
    p.add_argument("--lr-head", type=float, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--minio", action="store_true")
    p.add_argument("--no-registry", action="store_true")
    p.add_argument("--push-to-hub", type=str, default=None)
    p.add_argument("--resume-from", type=str, default=None)
    return p.parse_args()


def build_strategy(tcfg: dict):
    name = tcfg.get("strategy", "lora")
    cls = STRATEGIES.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Choose from {list(STRATEGIES)}")

    kwargs = {
        "lr": tcfg.get("lr_lora", 1e-4),
        "weight_decay": tcfg.get("weight_decay", 0.01),
        "warmup_steps": tcfg.get("warmup_steps", 100),
        "scheduler": tcfg.get("scheduler", "cosine"),
        "ema_decay": tcfg.get("ema_decay", 0.0),
        "grad_clip": tcfg.get("grad_clip", 1.0),
        "accumulation_steps": tcfg.get("grad_accum", 4),
        "amp": tcfg.get("amp", False),
        "early_stopping_patience": tcfg.get("early_stopping_patience", 0),
        "loss_fn": tcfg.get("loss_fn", "mse"),
    }
    if name == "lora":
        kwargs.update({
            "rank": tcfg.get("rank", 8),
            "alpha": tcfg.get("alpha", tcfg.get("rank", 8) * 2),
            "lr_lora": tcfg.get("lr_lora", 1e-4),
            "lr_head": tcfg.get("lr_head", 1e-3),
        })
        del kwargs["lr"]
    return cls(**kwargs)


def main():
    args = parse_args()

    config_path = args.config or DEFAULT_CONFIG
    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, args)

    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    ocfg = cfg["output"]
    scfg = cfg.get("storage", {})

    print(f"Config: {config_path}\n")

    t0 = time.time()

    # ── 1. Data ──────────────────────────────────────────────────────

    cache_dir = dcfg.get("cache_dir", "./data/kinases")
    collection = dcfg.get("collection", "kinases_human")
    max_structures = dcfg.get("max_structures", 200)
    resolution_max = dcfg.get("resolution_max", 2.5)
    storage = None

    print(f"[1/7] Searching RCSB for {collection} (max {max_structures})...")
    pdb_ids = search_collection(
        collection,
        max_structures=max_structures,
        resolution_max=resolution_max,
    )
    print(f"      {len(pdb_ids)} IDs found")

    if scfg.get("minio"):
        storage = MinioStorage.from_env()
        storage.ensure_bucket()
        print(f"      Syncing from MinIO → {cache_dir}")
        found, missing = storage.sync_ids_to_local(pdb_ids, collection, cache_dir)
        print(f"      {len(found)} from MinIO, {len(missing)} need RCSB")
        if missing:
            PDBFetcher(cache_dir=cache_dir).fetch(missing)
            print(f"      Downloaded {len(missing)} from RCSB")
            uploaded = storage.sync_to_remote(cache_dir, collection)
            if uploaded:
                print(f"      Uploaded {uploaded} new files to MinIO")
    else:
        print(f"      Fetching → {cache_dir}")
        fetch_collection(
            collection,
            max_structures=max_structures,
            resolution_max=resolution_max,
            cache_dir=cache_dir,
        )

    paths = [
        str(Path(cache_dir) / f"{pid.strip().lower()}.cif")
        for pid in pdb_ids
        if (Path(cache_dir) / f"{pid.strip().lower()}.cif").exists()
    ]
    print(f"      {len(paths)} structures ready")

    registry = None
    if scfg.get("registry", True):
        registry = ExperimentRegistry(storage=storage)
        registry.start_run(
            name=f"{collection}-{tcfg.get('strategy', 'lora')}",
            tags=[tcfg.get("strategy", "lora"), mcfg.get("head", "structure"), collection],
            config=cfg,
        )
    if registry:
        registry.log_pdb_refs(paths)

    # ── 2. Model ─────────────────────────────────────────────────────

    device = mcfg.get("device", "cuda")
    head = mcfg.get("head", "structure")
    head_config = mcfg.get("head_config") or {}

    print(f"[2/7] Loading {mcfg.get('backend', 'openfold')} on {device}...")
    model = MolfunStructureModel.from_pretrained(
        mcfg.get("backend", "openfold"),
        device=device,
        head=head,
        head_config=head_config if head_config else None,
    )
    summary = model.summary()
    print(f"      Total: {summary['adapter']['total']:,}  Trainable: {summary['adapter']['trainable']:,}")

    # ── 3. Dataset ───────────────────────────────────────────────────

    max_seq_len = dcfg.get("max_seq_len", 256)
    print(f"[3/7] Building dataset (seq_len={max_seq_len})...")

    featurizer = OpenFoldFeaturizer(
        config=model.adapter.model.config,
        max_seq_len=max_seq_len,
    )
    dataset = StructureDataset(pdb_paths=paths, featurizer=featurizer)
    train_ds, val_ds, test_ds = DataSplitter.random(
        dataset,
        val_frac=dcfg.get("val_frac", 0.1),
        test_frac=dcfg.get("test_frac", 0.1),
        seed=dcfg.get("split_seed", 42),
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)
    print(f"      Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── 4. Strategy ──────────────────────────────────────────────────

    strategy_name = tcfg.get("strategy", "lora")
    print(f"[4/7] Strategy: {strategy_name} (rank={tcfg.get('rank', '-')})")
    strategy = build_strategy(tcfg)

    # ── 5. Train ─────────────────────────────────────────────────────

    epochs = tcfg.get("epochs", 20)
    ckpt_dir = ocfg.get("checkpoint_dir", "./checkpoints")
    save_every = tcfg.get("save_every", 0)
    resume_from = tcfg.get("resume_from")

    print(f"[5/7] Training {epochs} epochs → checkpoints: {ckpt_dir}/")
    if resume_from:
        print(f"      Resuming from {resume_from}")

    history = model.fit(
        train_loader, val_loader,
        strategy=strategy,
        epochs=epochs,
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        tracker=registry,
        checkpoint_dir=ckpt_dir,
        save_every=save_every,
        resume_from=resume_from,
    )

    # ── 6. Checkpoints ───────────────────────────────────────────────

    print(f"\n[6/7] Checkpoints → {ckpt_dir}/")
    print(f"       best/ | last/" + (f" | every {save_every} epochs" if save_every else ""))

    # ── 7. Test ──────────────────────────────────────────────────────

    print("[7/7] Evaluating on test set...")
    model.adapter.eval()
    test_losses = []
    with torch.no_grad():
        for batch_data in test_loader:
            features = batch_data[0]
            features = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in features.items()}
            result = model.forward(features)
            if isinstance(result.get("preds"), torch.Tensor):
                test_losses.append(result["preds"].item())

    if test_losses:
        avg_loss = sum(test_losses) / len(test_losses)
        print(f"      Test loss: {avg_loss:.4f} ({len(test_losses)} samples)")
    else:
        print("      No test metrics computed")
        avg_loss = None

    # ── Hub ───────────────────────────────────────────────────────────

    hub_repo = ocfg.get("push_to_hub")
    if hub_repo:
        print(f"\nPushing → {hub_repo}...")
        url = model.push_to_hub(
            hub_repo,
            metrics={"test_loss": avg_loss} if avg_loss else None,
            dataset_name=f"{collection} ({len(paths)} structures)",
        )
        print(f"Uploaded: {url}")

    # ── End ───────────────────────────────────────────────────────────

    if registry:
        if avg_loss is not None:
            registry.log_metrics({"test_loss": avg_loss})
        registry.end_run()
        print(f"\nRegistry: {registry.run_dir}")
        if registry.remote_uri:
            print(f"Remote:   {registry.remote_uri}")

    elapsed = time.time() - t0
    print(f"\nDone in {int(elapsed // 60)}m {int(elapsed % 60)}s.")
    print(f"Checkpoint: {ckpt_dir}/")


if __name__ == "__main__":
    main()
