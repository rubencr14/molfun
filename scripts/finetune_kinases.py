#!/usr/bin/env python
"""
End-to-end fine-tuning of OpenFold on human kinases.

Downloads 200 kinase structures from RCSB PDB, fine-tunes with LoRA,
evaluates on a held-out test set, and saves the checkpoint.

Requirements:
    pip install molfun[openfold]
    pip install git+https://github.com/aqlaboratory/openfold.git

Hardware: NVIDIA GPU with >= 16 GB VRAM (RTX 4090, 5090, A100, etc.)

Usage:
    python scripts/finetune_kinases.py
    python scripts/finetune_kinases.py --epochs 10 --max-structures 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from molfun.models import MolfunStructureModel
from molfun.backends.openfold import OpenFoldFeaturizer
from molfun.data.collections import fetch_collection
from molfun.data.datasets import StructureDataset
from molfun.data.splits import DataSplitter
from molfun.training import LoRAFinetune
from molfun.storage import MinioStorage
from molfun.tracking import ExperimentRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune OpenFold on human kinases")
    p.add_argument("--max-structures", type=int, default=200,
                    help="Max kinase structures to download (default: 200)")
    p.add_argument("--max-seq-len", type=int, default=256,
                    help="Max sequence length for featurizer (default: 256)")
    p.add_argument("--epochs", type=int, default=20,
                    help="Training epochs (default: 20)")
    p.add_argument("--rank", type=int, default=8,
                    help="LoRA rank (default: 8)")
    p.add_argument("--lr-lora", type=float, default=1e-4,
                    help="LoRA learning rate (default: 1e-4)")
    p.add_argument("--lr-head", type=float, default=1e-3,
                    help="Head learning rate (default: 1e-3)")
    p.add_argument("--grad-accum", type=int, default=4,
                    help="Gradient accumulation steps (default: 4)")
    p.add_argument("--output-dir", type=str, default="./checkpoints/kinase_lora",
                    help="Checkpoint output directory")
    p.add_argument("--data-dir", type=str, default="./data/kinases",
                    help="PDB download cache directory")
    p.add_argument("--device", type=str, default="cuda",
                    help="Device: cuda or cpu (default: cuda)")
    p.add_argument("--minio", action="store_true",
                    help="Use MinIO storage (reads MINIO_* env vars)")
    p.add_argument("--no-registry", action="store_true",
                    help="Disable experiment registry")
    p.add_argument("--push-to-hub", type=str, default=None,
                    help="HuggingFace Hub repo ID (e.g. user/kinase-lora-v1)")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # ── 1. Download kinase structures from RCSB ──────────────────────

    storage_opts: dict | None = None
    cache_dir = args.data_dir
    storage = None

    if args.minio:
        storage = MinioStorage.from_env()
        created = storage.ensure_bucket()
        cache_dir = storage.prefix("kinases")
        storage_opts = storage.storage_options
        bucket_status = "created" if created else "exists"
        print(f"[1/7] Fetching up to {args.max_structures} kinases → MinIO ({cache_dir})  [{bucket_status}]")
    else:
        print(f"[1/7] Fetching up to {args.max_structures} kinases → {cache_dir}")

    registry = ExperimentRegistry(storage=storage) if not args.no_registry else None
    if registry is not None:
        registry.start_run(
            name="kinase-lora",
            tags=["lora", "structure", "kinase"],
            config={
                "max_structures": args.max_structures,
                "max_seq_len": args.max_seq_len,
                "epochs": args.epochs,
                "rank": args.rank,
                "lr_lora": args.lr_lora,
                "lr_head": args.lr_head,
                "grad_accum": args.grad_accum,
            },
        )

    paths = fetch_collection(
        "kinases_human",
        max_structures=args.max_structures,
        resolution_max=2.5,
        cache_dir=cache_dir,
        storage_options=storage_opts,
    )
    print(f"      {len(paths)} structures ready")
    if registry is not None:
        registry.log_pdb_refs(paths)

    # ── 2. Load pretrained model ─────────────────────────────────────

    print(f"[2/7] Loading pretrained OpenFold model on {args.device}...")
    model = MolfunStructureModel.from_pretrained(
        "openfold",
        device=args.device,
        head="structure",
    )

    summary = model.summary()
    print(f"      Total params:     {summary['adapter']['total']:,}")
    print(f"      Trainable params: {summary['adapter']['trainable']:,}")

    # ── 3. Build dataset and split ───────────────────────────────────

    print(f"[3/7] Building dataset (max_seq_len={args.max_seq_len})...")
    featurizer = OpenFoldFeaturizer(
        config=model.adapter.model.config,
        max_seq_len=args.max_seq_len,
    )

    dataset = StructureDataset(pdb_paths=paths, featurizer=featurizer)
    train_ds, val_ds, test_ds = DataSplitter.random(dataset, val_frac=0.1, test_frac=0.1)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

    print(f"      Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── 4. Configure LoRA strategy ───────────────────────────────────

    print(f"[4/7] Setting up LoRA fine-tuning (rank={args.rank})...")
    strategy = LoRAFinetune(
        rank=args.rank,
        lr_lora=args.lr_lora,
        lr_head=args.lr_head,
        ema_decay=0.999,
        accumulation_steps=args.grad_accum,
        warmup_steps=100,
        amp=False,
    )

    # ── 5. Fine-tune ─────────────────────────────────────────────────

    print(f"[5/7] Fine-tuning for {args.epochs} epochs...")
    print(f"       {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>11}")
    print(f"       {'─'*5}  {'─'*11}  {'─'*11}")

    ckpt_path = registry.checkpoint_path if registry is not None else args.output_dir
    history = model.fit(
        train_loader,
        val_loader,
        strategy=strategy,
        epochs=args.epochs,
        gradient_checkpointing=True,
        tracker=registry,
    )

    for epoch in history:
        train_loss = f"{epoch['train_loss']:.4f}"
        val_loss = f"{epoch.get('val_loss', 'N/A'):>11}" if isinstance(
            epoch.get('val_loss'), float
        ) else f"{'N/A':>11}"
        if isinstance(epoch.get('val_loss'), float):
            val_loss = f"{epoch['val_loss']:.4f}"
        print(f"       {epoch['epoch']:5d}  {train_loss:>11}  {val_loss:>11}")

    # ── 6. Save checkpoint ───────────────────────────────────────────

    print(f"[6/7] Saving checkpoint to {ckpt_path}...")
    model.save(ckpt_path)
    print(f"      Saved: {ckpt_path}/")

    # ── 7. Evaluate on test set ──────────────────────────────────────

    print("[7/7] Evaluating on test set...")
    model.adapter.eval()
    test_losses = []
    with torch.no_grad():
        for batch_data in test_loader:
            features = batch_data[0]
            features = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                        for k, v in features.items()}
            result = model.forward(features)
            if isinstance(result.get("preds"), torch.Tensor):
                test_losses.append(result["preds"].item())

    if test_losses:
        avg_loss = sum(test_losses) / len(test_losses)
        print(f"      Test loss (avg): {avg_loss:.4f}")
        print(f"      Test samples:    {len(test_losses)}")
    else:
        print("      No test metrics computed (check head configuration)")

    # ── Optional: push to HuggingFace Hub ────────────────────────────

    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}...")
        url = model.push_to_hub(
            args.push_to_hub,
            metrics={"test_loss": avg_loss} if test_losses else None,
            dataset_name=f"kinases_human ({len(paths)} structures)",
        )
        print(f"Uploaded: {url}")

    # ── End run ──────────────────────────────────────────────────────

    if registry is not None:
        if test_losses:
            registry.log_metrics({"test_loss": sum(test_losses) / len(test_losses)})
        registry.end_run(status="completed")
        print(f"\nRegistry: {registry.run_dir}")
        if registry.remote_uri:
            print(f"Remote:   {registry.remote_uri}")

    # ── Done ─────────────────────────────────────────────────────────

    elapsed = time.time() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\nDone in {minutes}m {seconds}s.")
    print(f"Checkpoint: {ckpt_path}")
    print(f"\nTo load and use the model:")
    print(f'  model = MolfunStructureModel("openfold", device="cuda", head="structure")')
    print(f'  model.load("{ckpt_path}")')
    print(f'  output = model.predict("MKWVTFISLLLLFSSAYS...")')


if __name__ == "__main__":
    main()
