# OpenFold Fine-Tuning Guide

Production-ready fine-tuning of OpenFold (AlphaFold2) for binding affinity prediction using Molfun.

## Prerequisites

```bash
# GPU: NVIDIA with >= 16 GB VRAM (tested on RTX 5090, A100, V100)
# CUDA 12.x + PyTorch 2.x

# Install OpenFold
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
    pip install git+https://github.com/aqlaboratory/openfold.git

pip install dm-tree ml-collections biopython

# Download pre-trained weights (model_1_ptm, ~370 MB)
mkdir -p ~/.molfun/weights
wget -O ~/.molfun/weights/finetuning_ptm_2.pt \
    https://huggingface.co/nz/openfold/resolve/main/finetuning_ptm_2.pt
```

## Data Preparation

Molfun expects a CSV with at least two columns and a directory of structure files:

```
data/
├── pdbbind.csv          # pdb_id, affinity, [resolution, year, ...]
├── pdbs/                # PDB/mmCIF files named {pdb_id}.cif
│   ├── 1a4w.cif
│   ├── 3htb.cif
│   └── ...
└── features/            # (optional) pre-computed OpenFold .pkl features
    ├── 1a4w.pkl
    └── ...
```

**CSV format** (minimal):

```csv
pdb_id,affinity
1a4w,6.52
3htb,8.10
2xys,4.30
```

The `affinity` column is the target value (typically -log(Kd) or pKi).

**Pre-computed features** are strongly recommended for production runs. They contain the full OpenFold input tensor dict (MSA features, templates, atom14 maps, etc.) serialized as pickle. Without them, Molfun parses PDB files on the fly and extracts basic features, which is sufficient for head-only training but not for tasks requiring full structural representations.

## Quick Start

Verify your setup works with synthetic data:

```bash
PYTHONPATH=. python scripts/fine_tune.py \
    --demo \
    --strategy lora \
    --epochs 3 \
    --output runs/demo/
```

Expected output:

```
Demo mode: 6 train / 2 val (synthetic, seq_len=32)
Model: OpenFold (93,237,338 params)
Strategy: LoRAFinetune (lr=1.0e-04)

Training for 3 epochs...
  epoch   1  train_loss=2642.29  val_loss=9.87  lr=8.01e-04
  epoch   2  train_loss=1474.72  val_loss=0.70  lr=2.77e-04
  epoch   3  train_loss=1216.14  val_loss=0.29  lr=1.00e-09
```

## Fine-Tuning Strategies

Molfun provides four strategies, each suited to a different data regime. All strategies share a common training infrastructure: cosine/linear LR scheduler with warmup, EMA, gradient accumulation, AMP, gradient clipping, and early stopping.

### 1. Head-Only (< 100 samples)

Freeze the entire 93M-parameter trunk. Train only the AffinityHead (~50K params). Fastest, zero risk of catastrophic forgetting.

```bash
PYTHONPATH=. python scripts/fine_tune.py \
    --data data/pdbbind.csv \
    --pdbs data/pdbs/ \
    --strategy head_only \
    --lr 1e-3 \
    --epochs 50 \
    --ema 0.999 \
    --warmup 50 \
    --output runs/head_only/
```

**When to use**: very small datasets, quick baselines, sanity checks.

### 2. LoRA (100 – 5K samples) — Recommended default

Freeze the trunk, inject low-rank adapters into the Evoformer attention layers. Trains ~600K LoRA params + head, keeping 99.3% of the model frozen.

```bash
PYTHONPATH=. python scripts/fine_tune.py \
    --data data/pdbbind.csv \
    --pdbs data/pdbs/ \
    --strategy lora \
    --lora-rank 8 \
    --lr 1e-4 \
    --epochs 30 \
    --accum 8 \
    --ema 0.999 \
    --warmup 200 \
    --output runs/lora/
```

**Key parameters**:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--lora-rank` | 8 | Higher = more capacity. 4 for small data, 16 for larger. |
| `--lr` | 1e-4 | LoRA LR. Head gets 10x this automatically. |
| `--accum` | 8 | Effective batch size = batch_size × accum. |
| `--ema` | 0.999 | EMA decay. Critical for small datasets. 0 to disable. |

LoRA targets `linear_q` and `linear_v` in all 48 Evoformer blocks by default. For more aggressive adaptation, add `linear_k` and `linear_o` in the Python API:

```python
from molfun.training import LoRAFinetune

strategy = LoRAFinetune(
    rank=16,
    target_modules=["linear_q", "linear_k", "linear_v", "linear_o"],
    lr_lora=5e-5,
    lr_head=5e-4,
    warmup_steps=500,
    ema_decay=0.999,
    accumulation_steps=8,
    early_stopping_patience=5,
)
```

### 3. Partial Fine-Tuning (1K – 10K samples)

Freeze early Evoformer blocks, unfreeze the last N blocks + structure module + head. Middle ground: the trunk adapts its task-specific layers while early (general) blocks stay fixed.

```bash
PYTHONPATH=. python scripts/fine_tune.py \
    --data data/pdbbind.csv \
    --pdbs data/pdbs/ \
    --strategy partial \
    --unfreeze-blocks 6 \
    --lr 1e-5 \
    --epochs 20 \
    --accum 8 \
    --ema 0.999 \
    --warmup 500 \
    --output runs/partial/
```

**Key parameters**:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--unfreeze-blocks` | 4 | Number of Evoformer blocks to unfreeze (from the end). OpenFold has 48. |
| `--lr` | 1e-5 | Trunk LR. Head gets 10x. Use lower LR than LoRA. |

Python API for finer control:

```python
from molfun.training import PartialFinetune

strategy = PartialFinetune(
    unfreeze_last_n=8,
    unfreeze_structure_module=True,
    lr_trunk=1e-5,
    lr_head=1e-3,
    warmup_steps=500,
    ema_decay=0.999,
    scheduler="cosine",
    grad_clip=0.5,
)
```

### 4. Full Fine-Tuning (> 10K samples)

Unfreeze everything. Layer-wise LR decay ensures early layers (general protein knowledge) change slowly while later layers (task-specific) adapt faster.

```bash
PYTHONPATH=. python scripts/fine_tune.py \
    --data data/pdbbind.csv \
    --pdbs data/pdbs/ \
    --strategy full \
    --lr 1e-5 \
    --epochs 10 \
    --accum 16 \
    --ema 0.999 \
    --warmup 1000 \
    --output runs/full/
```

**Warning**: Full fine-tuning requires substantial data. With < 5K samples you will almost certainly overfit, even with EMA and weight decay.

Python API:

```python
from molfun.training import FullFinetune

strategy = FullFinetune(
    lr=1e-5,
    lr_head=1e-3,
    layer_lr_decay=0.95,         # block_0 gets lr * 0.95^47, block_47 gets lr
    lr_embedder=1e-7,            # input embedder barely moves
    lr_structure_module=1e-5,
    warmup_steps=1000,
    ema_decay=0.999,
    accumulation_steps=16,
    weight_decay=0.01,
    early_stopping_patience=3,
)
```

## Strategy Selection Decision Tree

```
Dataset size?
│
├── < 100 samples ──────────── head_only
├── 100 – 5K samples ──────── lora (rank 4–16)
├── 1K – 10K samples ──────── partial (last 4–8 blocks)
└── > 10K samples ──────────── full (with LR decay + EMA)
```

If unsure, start with **LoRA rank=8**. It is the best risk/reward trade-off for most protein affinity datasets.

## Training Infrastructure

All strategies share these features via `FinetuneStrategy`:

| Feature | Flag | Default | Purpose |
|---------|------|---------|---------|
| LR Scheduler | `scheduler` | `"cosine"` | `"cosine"`, `"linear"`, or `"constant"` after warmup |
| Warmup | `warmup_steps` | 0 | Linear LR warmup from 0 to target |
| EMA | `ema_decay` | 0.0 | Exponential Moving Average. Set to 0.999 for production. Uses EMA weights during validation. |
| Gradient Accumulation | `accumulation_steps` | 1 | Simulates larger batch sizes. Effective batch = batch_size × accum |
| Gradient Clipping | `grad_clip` | 1.0 | Max gradient norm. 0 to disable. |
| Mixed Precision | `amp` | True | FP16 forward pass. Disable if you see NaN with short sequences. |
| Early Stopping | `early_stopping_patience` | 0 | Stop after N epochs without val_loss improvement. 0 = disabled. |
| Weight Decay | `weight_decay` | 0.01 | AdamW L2 regularization |

## Production Recommendations

### Hyperparameter Guidelines

```
# LoRA (most common case)
rank:       8            # 4 if dataset < 200, 16 if > 2K
lr_lora:    1e-4         # scale down for larger rank
lr_head:    1e-3         # 10x the LoRA LR
warmup:     10% of total steps
ema_decay:  0.999        # always use EMA for small datasets
accum:      8            # effective batch ~8 structures
epochs:     20-50        # with early stopping patience 5
grad_clip:  1.0
scheduler:  cosine
```

### Avoiding Common Pitfalls

1. **AMP + short sequences**: OpenFold's PTM score can produce NaN with float16 and sequences < 50 residues. Disable AMP for small synthetic tests, enable for production with real-length proteins.

2. **Batch size**: OpenFold is memory-hungry. Use `--batch-size 1 --accum 8` for effective batch 8 on a single GPU. A100 80GB can handle batch_size=2 for sequences up to ~300 residues.

3. **EMA is essential**: For datasets < 5K, EMA (0.999) acts as implicit ensembling and significantly stabilizes predictions. The training loop uses EMA weights for validation automatically.

4. **Learning rate ratio**: Always set `lr_head ≈ 10× lr_trunk`. The randomly initialized head needs to converge faster than the pre-trained trunk.

5. **Sequence length variance**: Proteins in PDBbind range from 50 to 1000+ residues. Gradient accumulation handles this naturally since each sample is processed independently before accumulation.

## Checkpoint Structure

After training, `model.save(path)` produces:

```
checkpoint/
├── meta.pt           # model name + strategy config (JSON-serializable)
├── peft_adapter      # LoRA weights (only for LoRA strategy)
└── head.pt           # AffinityHead state dict
```

### Loading a Checkpoint

```python
from openfold.config import model_config
from molfun.models.structure import MolfunStructureModel
from molfun.training import LoRAFinetune

# Rebuild model with same architecture
model = MolfunStructureModel(
    "openfold",
    config=model_config("model_1_ptm"),
    weights="~/.molfun/weights/finetuning_ptm_2.pt",
    device="cuda",
    head="affinity",
    head_config={"single_dim": 384, "hidden_dim": 128},
)

# Re-apply same strategy (needed to inject LoRA layers)
strategy = LoRAFinetune(rank=8, target_modules=["linear_q", "linear_v"], use_hf=False)
strategy.setup(model)

# Load fine-tuned weights
model.load("runs/lora/checkpoint")
```

### Merging LoRA for Deployment

For production inference without LoRA overhead:

```python
model.merge()               # LoRA weights fused into base model
output = model.predict(batch)
model.unmerge()              # restore LoRA (optional, for continued training)
```

## Python API (Programmatic Usage)

For integration into larger pipelines, use the Python API directly instead of the CLI:

```python
from openfold.config import model_config
from torch.utils.data import DataLoader

from molfun.models.structure import MolfunStructureModel
from molfun.data import AffinityDataset, DataSplitter
from molfun.training import LoRAFinetune

# Data
dataset = AffinityDataset.from_csv("data/pdbbind.csv", "data/pdbs/")
train_ds, val_ds, test_ds = DataSplitter.random(dataset)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                          collate_fn=AffinityDataset.collate_fn)
val_loader = DataLoader(val_ds, batch_size=1,
                        collate_fn=AffinityDataset.collate_fn)

# Model
model = MolfunStructureModel(
    "openfold",
    config=model_config("model_1_ptm"),
    weights="~/.molfun/weights/finetuning_ptm_2.pt",
    device="cuda",
    head="affinity",
    head_config={"single_dim": 384, "hidden_dim": 128},
)

# Train
strategy = LoRAFinetune(
    rank=8,
    lr_lora=1e-4,
    lr_head=1e-3,
    warmup_steps=200,
    ema_decay=0.999,
    accumulation_steps=8,
    early_stopping_patience=5,
)

history = model.fit(train_loader, val_loader, strategy=strategy, epochs=30)

# Save
model.save("runs/production/checkpoint")

# Inference
model.merge()
prediction = model.predict(test_batch)
```

## Verifying Installation

Run the built-in test suite to validate everything works on your hardware:

```bash
# Quick mock tests (no GPU needed, ~5s)
python -m pytest tests/models/test_openfold_wrapper.py -v

# Real GPU tests with pre-trained weights (~40s)
python -m pytest tests/models/test_openfold_real.py -v -s

# All strategy tests on real OpenFold (~45s)
python -m pytest tests/training/test_strategies_real.py -v -s

# Full test suite
python -m pytest tests/ -v
```
