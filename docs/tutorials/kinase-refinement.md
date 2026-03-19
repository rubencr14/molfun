---
title: Kinase Structure Refinement
---

# Kinase Structure Refinement

<span class="badge badge-intermediate">Intermediate</span> &nbsp; ~30 min

Refine predicted kinase structures using domain-specific data from Molfun's built-in
`kinases_human` collection. This tutorial uses **PartialFinetune** to unfreeze only the
last few trunk blocks, combined with **FAPE loss** for structure-aware training.

---

## What You Will Learn

- Fetch curated protein collections with `fetch_collection()`
- Use `PartialFinetune` to selectively unfreeze trunk blocks
- Train with FAPE (Frame Aligned Point Error) loss
- Evaluate structural quality with GDT-TS and lDDT metrics

## Prerequisites

- Molfun installed with data extras: `pip install molfun[data]`
- GPU recommended for structure prediction training

---

## Overview

```mermaid
graph LR
    A["fetch_collection<br/><small>kinases_human</small>"] --> B["StructureDataset"]
    B --> C["DataLoader"]
    C --> D["MolfunStructureModel<br/><small>PartialFinetune</small>"]
    D --> E["FAPE Loss"]
    E --> F["Refined Structures"]

    style A fill:#0d9488,stroke:#0f766e,color:#ffffff
    style D fill:#7c3aed,stroke:#6d28d9,color:#ffffff
    style E fill:#d97706,stroke:#b45309,color:#ffffff
    style F fill:#16a34a,stroke:#15803d,color:#ffffff
```

---

## Step 1: Fetch the Kinase Collection

Molfun ships with curated protein collections. The `kinases_human` collection contains
human kinase sequences paired with experimentally determined structures.

```python
from molfun.data import fetch_collection, StructureDataset, DataSplitter
from torch.utils.data import DataLoader

# Fetch the curated kinase collection
kinases = fetch_collection("kinases_human")

print(f"Fetched {len(kinases)} kinase structures")
print(f"Example: {kinases[0].pdb_id} ({kinases[0].name})")
print(f"  Sequence length: {len(kinases[0].sequence)} residues")
```

!!! info "Available collections"

    Molfun provides several built-in collections:

    - `kinases_human` --- Human protein kinases (~500 structures)
    - `gpcrs_human` --- G protein-coupled receptors
    - `antibodies_therapeutic` --- Therapeutic antibody structures

    Use `fetch_collection("name")` to download and cache them locally.

---

## Step 2: Prepare the Dataset

```python
# Create structure dataset with experimental coordinates as targets
dataset = StructureDataset(
    sequences=[k.sequence for k in kinases],
    structures=[k.coordinates for k in kinases],  # Target atom positions
    max_length=600,
)

# Split 80/20
splitter = DataSplitter(test_size=0.2, random_state=42)
train_ds, val_ds = splitter.split(dataset)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)  # (1)!
val_loader = DataLoader(val_ds, batch_size=2)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
```

1. Batch size is small because structure prediction requires significant GPU memory
   per sample, especially for longer sequences.

---

## Step 3: Configure PartialFinetune

`PartialFinetune` unfreezes the last N transformer blocks of the trunk while keeping
earlier layers frozen. This provides a middle ground between head-only training (fast but
limited) and full fine-tuning (powerful but expensive).

```python
from molfun import MolfunStructureModel
from molfun.training import PartialFinetune

model = MolfunStructureModel.from_pretrained(
    "openfold_v1",
    device="cuda",
)

strategy = PartialFinetune(
    n_unfrozen_blocks=4,  # Unfreeze last 4 Evoformer blocks
    lr=5e-5,              # Learning rate for unfrozen parameters
)
```

??? question "How many blocks should I unfreeze?"

    The right number depends on your task:

    | Blocks | Use case | GPU memory |
    |:------:|----------|:----------:|
    | 1--2   | Minor refinement, small dataset | Low |
    | 3--4   | Domain adaptation (recommended start) | Medium |
    | 6--8   | Significant distribution shift | High |
    | All    | Equivalent to full fine-tune | Maximum |

    Start with 4 blocks and increase if validation loss plateaus.

---

## Step 4: Train with FAPE Loss

FAPE (Frame Aligned Point Error) is the primary structure loss used by AlphaFold2. It
measures the error in predicted atom positions after aligning local reference frames,
making it rotationally invariant.

```python
from molfun.losses import LOSS_REGISTRY

# FAPE loss is registered by default
fape_loss = LOSS_REGISTRY["fape"]
print(f"Using loss: {fape_loss}")

model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    strategy=strategy,
    epochs=25,
    checkpoint_dir="checkpoints/kinase_partial",
)
```

!!! note "Loss selection"

    FAPE is the default loss for structure prediction tasks. If you want to combine
    losses, you can pass a list:

    ```python
    from molfun.losses import LOSS_REGISTRY

    losses = {
        "fape": LOSS_REGISTRY["fape"],
        "mse": LOSS_REGISTRY["mse"],
    }
    ```

---

## Step 5: Evaluate Structural Quality

After training, evaluate the refined structures against experimental references using
standard structural quality metrics.

```python
import torch
import numpy as np

model.eval()
gdt_scores = []
lddt_scores = []

with torch.no_grad():
    for batch in val_loader:
        output = model.predict(batch["sequence"])

        # Compare predicted coordinates to experimental targets
        for pred_coords, true_coords in zip(
            output.atom_positions, batch["structures"]
        ):
            # GDT-TS (Global Distance Test - Total Score)
            gdt = compute_gdt_ts(pred_coords, true_coords)
            gdt_scores.append(gdt)

            # lDDT (Local Distance Difference Test)
            lddt = output.plddt.mean().item()
            lddt_scores.append(lddt)

print(f"Mean GDT-TS: {np.mean(gdt_scores):.1f}")
print(f"Mean lDDT:   {np.mean(lddt_scores):.1f}")
```

### Visualize Improvement

Compare structural quality before and after fine-tuning:

```python
import matplotlib.pyplot as plt

# Assume we collected metrics for baseline and refined models
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# GDT-TS distribution
axes[0].hist(baseline_gdt, alpha=0.6, label="Baseline", bins=20)
axes[0].hist(gdt_scores, alpha=0.6, label="Refined", bins=20)
axes[0].set_xlabel("GDT-TS")
axes[0].set_ylabel("Count")
axes[0].legend()
axes[0].set_title("Structure Quality (GDT-TS)")

# Per-protein improvement
improvement = np.array(gdt_scores) - np.array(baseline_gdt)
axes[1].bar(range(len(improvement)), sorted(improvement, reverse=True))
axes[1].axhline(y=0, color="r", linestyle="--")
axes[1].set_xlabel("Protein index")
axes[1].set_ylabel("GDT-TS improvement")
axes[1].set_title("Per-Protein Improvement")

plt.tight_layout()
plt.savefig("kinase_refinement_results.png", dpi=150)
plt.show()
```

---

## Step 6: Save the Refined Model

```python
model.save("models/kinase_refined")

# Push to HuggingFace Hub for sharing
model.push_to_hub("your-username/kinase-structure-refined")
```

---

## Full Script

??? abstract "Complete runnable code"

    ```python
    """Kinase structure refinement with PartialFinetune."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from molfun import MolfunStructureModel
    from molfun.data import fetch_collection, StructureDataset, DataSplitter
    from molfun.training import PartialFinetune

    # ── Data ──────────────────────────────────────────────
    kinases = fetch_collection("kinases_human")

    dataset = StructureDataset(
        sequences=[k.sequence for k in kinases],
        structures=[k.coordinates for k in kinases],
        max_length=600,
    )

    splitter = DataSplitter(test_size=0.2, random_state=42)
    train_ds, val_ds = splitter.split(dataset)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)

    # ── Model ─────────────────────────────────────────────
    model = MolfunStructureModel.from_pretrained("openfold_v1", device="cuda")

    strategy = PartialFinetune(n_unfrozen_blocks=4, lr=5e-5)

    # ── Train ─────────────────────────────────────────────
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        strategy=strategy,
        epochs=25,
        checkpoint_dir="checkpoints/kinase_partial",
    )

    # ── Evaluate ──────────────────────────────────────────
    model.eval()
    lddt_scores = []
    with torch.no_grad():
        for batch in val_loader:
            output = model.predict(batch["sequence"])
            lddt_scores.append(output.plddt.mean().item())

    print(f"Mean lDDT: {np.mean(lddt_scores):.1f}")

    # ── Save ──────────────────────────────────────────────
    model.save("models/kinase_refined")
    ```

---

## Next Steps

- **Want to customize the architecture?** See [Custom Architectures](custom-architectures.md)
  to build models with different attention and structure modules.
- **Need a reproducible pipeline?** Wrap this workflow in a
  [YAML Pipeline](yaml-pipelines.md).
- **Comparing strategies?** Track all your runs with
  [Experiment Tracking](experiment-tracking.md).
