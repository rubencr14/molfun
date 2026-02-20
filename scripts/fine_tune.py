#!/usr/bin/env python
"""
Fine-tune OpenFold on a binding affinity dataset.

Usage (real data):
    PYTHONPATH=. python scripts/fine_tune.py \
        --data data/pdbbind.csv \
        --pdbs data/pdbs/ \
        --strategy lora \
        --output runs/lora_pdbbind/

Usage (demo with synthetic data):
    PYTHONPATH=. python scripts/fine_tune.py --demo \
        --strategy lora --epochs 3 --output runs/demo/
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from openfold.config import model_config

from molfun.models.structure import MolfunStructureModel
from molfun.training import (
    HeadOnlyFinetune,
    LoRAFinetune,
    PartialFinetune,
    FullFinetune,
)

STRATEGIES = {
    "head_only": lambda args: HeadOnlyFinetune(
        lr=args.lr,
        warmup_steps=args.warmup,
        ema_decay=args.ema,
        amp=args.amp,
    ),
    "lora": lambda args: LoRAFinetune(
        rank=args.lora_rank,
        alpha=args.lora_rank * 2.0,
        target_modules=["linear_q", "linear_v"],
        use_hf=False,
        lr_lora=args.lr,
        lr_head=args.lr * 10,
        warmup_steps=args.warmup,
        ema_decay=args.ema,
        accumulation_steps=args.accum,
        amp=args.amp,
    ),
    "partial": lambda args: PartialFinetune(
        unfreeze_last_n=args.unfreeze_blocks,
        unfreeze_structure_module=True,
        lr_trunk=args.lr,
        lr_head=args.lr * 10,
        warmup_steps=args.warmup,
        ema_decay=args.ema,
        accumulation_steps=args.accum,
        amp=args.amp,
    ),
    "full": lambda args: FullFinetune(
        lr=args.lr,
        lr_head=args.lr * 10,
        layer_lr_decay=0.95,
        warmup_steps=args.warmup,
        ema_decay=args.ema,
        accumulation_steps=args.accum,
        amp=args.amp,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune OpenFold for binding affinity")

    p.add_argument("--demo", action="store_true",
                    help="Run with synthetic data (no CSV/PDBs needed)")
    p.add_argument("--data", type=str, default=None,
                    help="CSV with pdb_id and affinity columns")
    p.add_argument("--pdbs", type=str, default=None,
                    help="Directory with PDB/mmCIF structure files")
    p.add_argument("--weights", type=str,
                    default=str(Path.home() / ".molfun/weights/finetuning_ptm_2.pt"),
                    help="Path to OpenFold pre-trained weights")
    p.add_argument("--features", type=str, default=None,
                    help="Pre-computed OpenFold feature pickles directory")
    p.add_argument("--output", type=str, default="runs/finetune/",
                    help="Output directory for checkpoints and metrics")

    p.add_argument("--strategy", choices=list(STRATEGIES), default="lora")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--demo-samples", type=int, default=8)
    p.add_argument("--demo-seq-len", type=int, default=32)

    p.add_argument("--split", choices=["random", "temporal", "identity"],
                    default="random")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if not args.demo and (args.data is None or args.pdbs is None):
        p.error("--data and --pdbs are required unless --demo is set")
    return args


def _build_demo_loaders(args):
    """Synthetic OpenFold-compatible data for quick validation."""
    from tests.models.test_openfold_real import DummyOpenFoldDataset, _collate

    n = args.demo_samples
    n_val = max(1, n // 4)
    n_train = n - n_val

    train_ds = DummyOpenFoldDataset(n_samples=n_train, seq_len=args.demo_seq_len, n_msa=4)
    val_ds = DummyOpenFoldDataset(n_samples=n_val, seq_len=args.demo_seq_len, n_msa=4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=_collate)

    return train_loader, val_loader, n_train, n_val


def _build_real_loaders(args):
    """Load real affinity data from CSV + PDB directory."""
    from molfun.data import AffinityDataset, DataSplitter

    dataset = AffinityDataset.from_csv(
        args.data, args.pdbs, features_dir=args.features,
    )
    train_ds, val_ds, _ = DataSplitter.random(
        dataset, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=AffinityDataset.collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        collate_fn=AffinityDataset.collate_fn, num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────
    if args.demo:
        train_loader, val_loader, n_train, n_val = _build_demo_loaders(args)
        print(f"Demo mode: {n_train} train / {n_val} val (synthetic, seq_len={args.demo_seq_len})")
    else:
        train_loader, val_loader, n_train, n_val = _build_real_loaders(args)
        print(f"Dataset: {n_train} train / {n_val} val")

    # ── Model ─────────────────────────────────────────────────────────
    config = model_config("model_1_ptm")
    model = MolfunStructureModel(
        "openfold",
        config=config,
        weights=args.weights,
        device="cuda",
        head="affinity",
        head_config={"single_dim": 384, "hidden_dim": 128},
    )
    total = model.summary()["adapter"]["total"]
    print(f"Model: OpenFold ({total:,} params)")

    # ── Strategy ──────────────────────────────────────────────────────
    strategy = STRATEGIES[args.strategy](args)
    desc = strategy.describe()
    print(f"Strategy: {desc['strategy']} (lr={desc['lr']:.1e})")

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    history = model.fit(
        train_loader, val_loader,
        strategy=strategy,
        epochs=args.epochs,
    )

    print()
    for h in history:
        line = f"  epoch {h['epoch']:3d}  train_loss={h['train_loss']:.4f}"
        if "val_loss" in h:
            line += f"  val_loss={h['val_loss']:.4f}"
        line += f"  lr={h['lr']:.2e}"
        print(line)

    # ── Save ──────────────────────────────────────────────────────────
    model.save(str(out / "checkpoint"))
    torch.save(history, out / "history.pt")
    print(f"\nCheckpoint saved to {out / 'checkpoint'}")
    print(f"History saved to {out / 'history.pt'}")


if __name__ == "__main__":
    main()
