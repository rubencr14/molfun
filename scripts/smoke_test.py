#!/usr/bin/env python
"""
Smoke test: end-to-end fine-tuning in ~15 minutes.

10 kinase structures, 3 epochs, seq_len=128.
Verifies the full pipeline runs without errors.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --minio          # sync run to MinIO
    python scripts/smoke_test.py --output-dir /tmp/smoke_ckpt
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings

# ── Silence third-party warnings ─────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.disable(logging.WARNING)

import torch
from torch.utils.data import DataLoader

from molfun.models import MolfunStructureModel
from molfun.backends.openfold import OpenFoldFeaturizer
from molfun.data.collections import fetch_collection
from molfun.data.datasets import StructureDataset
from molfun.data.splits import DataSplitter
from molfun.training import LoRAFinetune
from molfun.tracking import ExperimentRegistry


MAX_STRUCTURES = 10
MAX_SEQ_LEN    = 128
EPOCHS         = 3
RANK           = 4
ALPHA          = RANK * 2
LR_LORA        = 1e-4
LR_HEAD        = 1e-3
GRAD_ACCUM     = 2
WARMUP         = 10
DATA_DIR       = "./data/kinases"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--minio", action="store_true",
                   help="Sync experiment registry to MinIO (reads MINIO_* env vars)")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    print(f"Smoke test  |  {MAX_STRUCTURES} structures  |  seq_len={MAX_SEQ_LEN}  |  {EPOCHS} epochs  |  {DEVICE}")
    print()

    storage = None
    if args.minio:
        from molfun.storage import MinioStorage
        storage = MinioStorage.from_env()
        storage.ensure_bucket()
        print(f"MinIO: {storage.uri}")

    registry = ExperimentRegistry(storage=storage)
    registry.start_run(
        name="smoke-test",
        tags=["smoke", "lora", "kinase"],
        config={
            "max_structures": MAX_STRUCTURES,
            "max_seq_len": MAX_SEQ_LEN,
            "epochs": EPOCHS,
            "rank": RANK,
            "alpha": ALPHA,
            "lr_lora": LR_LORA,
            "lr_head": LR_HEAD,
        },
    )

    # 1. Structures
    print("[1/5] Fetching structures...")
    paths = fetch_collection(
        "kinases_human",
        max_structures=MAX_STRUCTURES,
        resolution_max=2.5,
        cache_dir=DATA_DIR,
    )
    print(f"      {len(paths)} structures ready")
    registry.log_pdb_refs(paths)

    # 2. Model
    print("[2/5] Loading model...")
    model = MolfunStructureModel.from_pretrained("openfold", device=DEVICE, head="structure")
    s = model.summary()
    print(f"      Total: {s['adapter']['total']:,}  Trainable: {s['adapter']['trainable']:,}")

    # 3. Dataset
    print("[3/5] Building dataset...")
    featurizer = OpenFoldFeaturizer(config=model.adapter.model.config, max_seq_len=MAX_SEQ_LEN)
    dataset = StructureDataset(pdb_paths=paths, featurizer=featurizer)
    train_ds, val_ds, _ = DataSplitter.random(dataset, val_frac=0.2, test_frac=0.0, seed=0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, num_workers=0)
    print(f"      Train: {len(train_ds)} | Val: {len(val_ds)}")

    # 4. Strategy
    print("[4/5] Setting up LoRA...")
    strategy = LoRAFinetune(
        rank=RANK,
        alpha=ALPHA,
        lr_lora=LR_LORA,
        lr_head=LR_HEAD,
        accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP,
        amp=False,
        ema_decay=0.0,
    )

    # 5. Train
    print(f"[5/5] Training {EPOCHS} epochs...")
    print(f"  {'Epoch':>5}  {'Train':>10}  {'Val':>10}  {'Time':>8}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}")

    history = model.fit(
        train_loader, val_loader,
        strategy=strategy,
        epochs=EPOCHS,
        gradient_checkpointing=True,
        tracker=registry,
    )

    for r in history:
        val = f"{r['val_loss']:.4f}" if isinstance(r.get("val_loss"), float) else "  —   "
        print(f"  {r['epoch']:5d}  {r['train_loss']:10.4f}  {val:>10}  {r.get('epoch_time', ''):>8}")

    ckpt_path = registry.checkpoint_path or OUTPUT_DIR
    model.save(ckpt_path)
    registry.end_run(status="completed")

    elapsed = time.time() - t0
    print(f"\nDone in {int(elapsed//60)}m {int(elapsed%60)}s  |  checkpoint: {ckpt_path}")
    print(f"Registry (local):  {registry.run_dir}")
    if registry.remote_uri:
        print(f"Registry (remote): {registry.remote_uri}")
    print("Pipeline OK" if history else "No history returned")


if __name__ == "__main__":
    main()
