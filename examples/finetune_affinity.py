#!/usr/bin/env python
"""
Fine-tune OpenFold on protein-ligand binding affinity (PDBbind-style).

This example shows the full pipeline:
  1. Fetch structures from RCSB PDB
  2. Load affinity labels from a CSV
  3. Split by sequence identity (avoids data leakage)
  4. Fine-tune with LoRA (parameter-efficient, works on 16GB GPU)
  5. Evaluate on held-out test set
  6. Push the model to Hugging Face Hub

Requires:
  pip install molfun[openfold]
  # and OpenFold weights at ~/.molfun/weights/finetuning_ptm_2.pt

Typical runtime: ~15 min on a single A100 for 100 structures / 20 epochs.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from molfun.data.sources.pdb import PDBFetcher
from molfun.data.datasets.structure import StructureDataset, collate_structure_batch
from molfun.data.sources.msa import MSAProvider
from molfun.data.splits import DataSplitter
from molfun.models.structure import MolfunStructureModel
from molfun.training import LoRAFinetune
from molfun.tracking import ConsoleTracker

# ── 1. Fetch structures ──────────────────────────────────────────────

fetcher = PDBFetcher(cache_dir="data/structures", fmt="cif")

# Kinase family — real PDBbind-like set
pdb_ids = [
    "3hb5", "4yne", "3rcd", "4agc", "3eqr",
    "2xyn", "3ge7", "4bcp", "3poz", "4agd",
    "3d83", "4ase", "3eml", "4bcn", "3lfa",
    "3g2y", "4bco", "3myg", "4ags", "3hmm",
]
paths = fetcher.fetch(pdb_ids)
print(f"Fetched {len(paths)} structures")

# ── 2. Load labels ───────────────────────────────────────────────────

# In a real project, load from PDBbind index or your own CSV:
#   from molfun.data.sources.affinity import AffinitySource
#   labels = AffinitySource.from_pdbbind_index("INDEX_refined_data.2020")
#
# For this example, synthetic labels:
labels = {pid: 5.0 + i * 0.3 for i, pid in enumerate(pdb_ids)}

# ── 3. Build dataset + split ─────────────────────────────────────────

msa = MSAProvider("single")  # use "colabfold" for real MSAs

dataset = StructureDataset(
    pdb_paths=paths,
    labels=labels,
    msa_provider=msa,
    max_seq_len=256,
)

# Option A: Random split (fast, for quick prototyping)
# train_ds, val_ds, test_ds = DataSplitter.random(dataset, val_frac=0.15, test_frac=0.15)

# Option B: Split by sequence identity — avoids data leakage from homologs
# Requires MMseqs2 installed; falls back to random if missing
train_ds, val_ds, test_ds = DataSplitter.by_sequence_identity(
    dataset, threshold=0.3, val_frac=0.15, test_frac=0.15,
)

# Option C: Temporal split (if you have deposition years)
# train_ds, val_ds, test_ds = DataSplitter.temporal(dataset, years, val_cutoff=2019, test_cutoff=2020)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_structure_batch)
val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_structure_batch)
test_loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_structure_batch)

print(f"Split: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

# ── 4. Fine-tune with LoRA ───────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"

from openfold.config import model_config
config = model_config("model_1_ptm")

model = MolfunStructureModel(
    "openfold",
    config=config,
    weights=str(Path.home() / ".molfun/weights/finetuning_ptm_2.pt"),
    device=device,
    head="affinity",
    head_config={"single_dim": 384, "hidden_dim": 128},
)

strategy = LoRAFinetune(
    rank=8,
    alpha=16.0,
    lr_lora=2e-4,
    lr_head=1e-3,
    warmup_steps=50,
    ema_decay=0.999,
    early_stopping_patience=5,
    loss_fn="mse",
)

tracker = ConsoleTracker(verbose=True)

history = strategy.fit(
    model, train_loader, val_loader,
    epochs=20, tracker=tracker,
)

# ── 5. Evaluate ──────────────────────────────────────────────────────

print("\nTest set evaluation:")
model.adapter.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for batch_data in test_loader:
        from molfun.helpers.training import unpack_batch, to_device
        batch, targets, mask = unpack_batch(batch_data)
        batch = to_device(batch, device)
        result = model.forward(batch, mask=mask)
        if "preds" in result and targets is not None:
            all_preds.append(result["preds"].cpu())
            all_targets.append(targets)

if all_preds:
    preds = torch.cat(all_preds).squeeze()
    targets = torch.cat(all_targets).squeeze()
    mae = (preds - targets).abs().mean().item()
    print(f"  MAE:  {mae:.4f}")

# ── 6. Save / push to Hub ────────────────────────────────────────────

model.save("runs/affinity_lora/checkpoint")
print(f"Checkpoint saved to runs/affinity_lora/checkpoint")

# Uncomment to push to Hugging Face Hub:
# url = model.push_to_hub(
#     "your-username/kinase-affinity-lora",
#     metrics={"test_mae": mae},
#     dataset_name="PDBbind-kinases",
# )
# print(f"Pushed to {url}")
