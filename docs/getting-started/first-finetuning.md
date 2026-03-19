---
title: First Fine-Tuning
---

# Your First Fine-Tuning

This tutorial walks through a complete LoRA fine-tuning workflow: preparing a dataset,
choosing a strategy, training with experiment tracking, saving the model, and evaluating
the results.

!!! info "Time estimate: 15 minutes (reading) + training time"

    A GPU is recommended for training but not strictly required. The examples use small
    datasets that can run on CPU for demonstration purposes.

---

## Overview

Fine-tuning adapts a pretrained model to your specific task --- for example, improving
structure prediction accuracy on a protein family or predicting binding affinity.
Molfun provides four strategies with different trade-offs:

| Strategy | Class | Trainable Params | Best For | Dataset Size |
|----------|-------|:----------------:|----------|:------------:|
| **Head-Only** | `HeadOnlyFinetune` | ~1--2% | New prediction heads, limited compute | 50--500 |
| **LoRA** | `LoRAFinetune` | ~0.5--1% | Most fine-tuning tasks, limited data | 100--5,000 |
| **Partial** | `PartialFinetune` | ~10--30% | Domain adaptation, moderate data | 1,000--10,000 |
| **Full** | `FullFinetune` | 100% | Large datasets, maximum accuracy | 10,000+ |

!!! tip "Start with LoRA"

    LoRA offers the best balance of parameter efficiency and performance. Start here unless
    you have a strong reason to choose another strategy.

---

## Step 1: Prepare Your Dataset

Molfun works with standard PyTorch `Dataset` and `DataLoader` objects. Your dataset should
return dictionaries with the keys your model expects.

### Structure Fine-Tuning

For structure prediction tasks, each sample needs a sequence and target coordinates:

```python
import torch
from torch.utils.data import Dataset, DataLoader

class StructureDataset(Dataset):
    """Minimal dataset for structure fine-tuning."""

    def __init__(self, sequences: list[str], coordinates: list[torch.Tensor]):
        self.sequences = sequences
        self.coordinates = coordinates  # each: (N_residues, 37, 3)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "target_coords": self.coordinates[idx],
        }
```

### Affinity Prediction

For binding affinity tasks, include the ligand SMILES and a scalar target:

```python
class AffinityDataset(Dataset):
    """Minimal dataset for affinity fine-tuning."""

    def __init__(self, sequences, ligands, affinities):
        self.sequences = sequences
        self.ligands = ligands          # SMILES strings
        self.affinities = affinities    # binding affinity in kcal/mol

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "ligand_smiles": self.ligands[idx],
            "target_affinity": self.affinities[idx],
        }
```

---

## Step 2: Create DataLoaders

Split your data into training and validation sets, then wrap them in DataLoaders:

```python
from torch.utils.data import random_split, DataLoader

# Assume `dataset` is your full dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=2,       # (1)!
    shuffle=True,
    num_workers=4,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=4,
)
```

1. Protein structure models are memory-intensive. Start with a small batch size
   (1--4) and increase if your GPU memory allows.

!!! warning "Batch size and memory"

    A single OpenFold forward pass on a 256-residue protein uses approximately 4--6 GB
    of GPU memory. With gradients and optimizer state, budget ~12 GB per sample for
    training. Use gradient accumulation if you need a larger effective batch size.

---

## Step 3: Choose a Strategy

Import and configure the LoRA strategy:

```python
from molfun.training import LoRAFinetune

strategy = LoRAFinetune(
    rank=8,          # (1)!
    alpha=16.0,      # (2)!
    lr_lora=1e-4,    # (3)!
    lr_head=1e-3,    # (4)!
)
```

1. The rank of the low-rank decomposition. Higher rank = more parameters = more capacity.
   Typical values: 4, 8, 16.
2. LoRA scaling factor. A common rule of thumb is `alpha = 2 * rank`.
3. Learning rate for the LoRA adapter weights. Keep this small --- the pretrained
   representations are already good.
4. Learning rate for the prediction head. The head is trained from scratch (or near-scratch),
   so it can tolerate a higher learning rate.

### Other Strategies

=== "Head-Only"

    Freeze the entire trunk and train only the prediction head. The fastest option,
    ideal when you have very little data or want to attach a new task head.

    ```python
    from molfun.training import HeadOnlyFinetune

    strategy = HeadOnlyFinetune(
        lr=1e-3,
        weight_decay=1e-2,
    )
    ```

=== "Partial"

    Unfreeze the last N transformer blocks while keeping earlier layers frozen.
    A middle ground between LoRA and full fine-tuning.

    ```python
    from molfun.training import PartialFinetune

    strategy = PartialFinetune(
        n_unfrozen_blocks=4,   # unfreeze last 4 Evoformer blocks
        lr=5e-5,
    )
    ```

=== "Full"

    All parameters are trainable. Use when you have a large dataset and sufficient compute.

    ```python
    from molfun.training import FullFinetune

    strategy = FullFinetune(
        lr=1e-5,
        weight_decay=1e-2,
        warmup_steps=500,
        lr_decay_factor=0.95,
    )
    ```

---

## Step 4: Train the Model

Load a pretrained model and call `model.fit()`:

```python
from molfun import MolfunStructureModel

# Load pretrained weights
model = MolfunStructureModel.from_pretrained("openfold", device="cuda")

# Fine-tune
metrics = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,          # (1)!
    strategy=strategy,
    epochs=10,
    tracker="wandb",                # (2)!
    checkpoint_dir="./checkpoints", # (3)!
)
```

1. Optional but recommended. Validation metrics are computed at the end of each epoch.
2. Experiment tracker. Options: `"wandb"`, `"comet"`, `"mlflow"`, `"langfuse"`, `"console"`,
   or `None`. You can also pass a tracker instance for custom configuration.
3. Checkpoints are saved at the end of each epoch. Use `resume_from` to continue
   training from a checkpoint.

`model.fit()` returns a list of dictionaries, one per epoch, containing training and
validation metrics:

```python
for epoch_metrics in metrics:
    print(
        f"Epoch {epoch_metrics['epoch']}: "
        f"train_loss={epoch_metrics['train_loss']:.4f}, "
        f"val_loss={epoch_metrics.get('val_loss', 'N/A')}"
    )
```

!!! tip "Resuming training"

    If training is interrupted, resume from the latest checkpoint:

    ```python
    metrics = model.fit(
        train_loader=train_loader,
        strategy=strategy,
        epochs=10,
        checkpoint_dir="./checkpoints",
        resume_from="./checkpoints/epoch_5.pt",
    )
    ```

---

## Step 5: Save and Evaluate

### Save the Model

```python
# Save the full model (base + LoRA adapters)
model.save("my_finetuned_model")

# Later, load it back
model = MolfunStructureModel.load("my_finetuned_model")
```

### Merge LoRA Weights

For deployment, merge the LoRA adapters into the base weights to eliminate the inference
overhead:

```python
model.merge()    # merges LoRA weights into the base model
model.save("my_merged_model")

# If you need to undo the merge (e.g., to continue training):
model.unmerge()
```

### Export for Production

```python
# Export to ONNX
model.export_onnx("model.onnx", seq_len=256)

# Export to TorchScript
model.export_torchscript("model.pt", seq_len=256)
```

### Push to HuggingFace Hub

Share your fine-tuned model with the community:

```python
model.push_to_hub(
    repo_id="your-username/my-protein-model",
    token="hf_...",
    private=False,
    metrics={"val_loss": 0.042, "val_rmsd": 1.23},
)

# Others can then load it with:
model = MolfunStructureModel.from_hub("your-username/my-protein-model")
```

### Evaluate

Run inference on a held-out test set to measure performance:

```python
model.predict(test_sequence)
# Compare output.structure_coords against ground truth
```

---

## Complete Example

Here is the full workflow in one script:

```python
import torch
from torch.utils.data import DataLoader, random_split
from molfun import MolfunStructureModel
from molfun.training import LoRAFinetune

# --- 1. Dataset ---
# Replace with your actual dataset
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# --- 2. Model ---
model = MolfunStructureModel.from_pretrained("openfold", device="cuda")

# --- 3. Strategy ---
strategy = LoRAFinetune(rank=8, alpha=16.0, lr_lora=1e-4, lr_head=1e-3)

# --- 4. Train ---
metrics = model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    strategy=strategy,
    epochs=10,
    tracker="wandb",
    checkpoint_dir="./checkpoints",
)

# --- 5. Save & export ---
model.merge()
model.save("finetuned_model")
model.export_onnx("finetuned_model.onnx", seq_len=256)

print(f"Final val_loss: {metrics[-1].get('val_loss', 'N/A')}")
```

---

## Strategy Comparison

Use this table to choose the right strategy for your scenario:

| | Head-Only | LoRA | Partial | Full |
|---|:---:|:---:|:---:|:---:|
| **Trainable params** | ~1--2% | ~0.5--1% | ~10--30% | 100% |
| **Training speed** | Fastest | Fast | Moderate | Slowest |
| **GPU memory** | Low | Low | Medium | High |
| **Min dataset size** | ~50 | ~100 | ~1,000 | ~10,000 |
| **Risk of overfitting** | Low | Low | Medium | High |
| **Best use case** | New task head | General fine-tuning | Domain adaptation | Large-scale retraining |
| **Supports merge** | N/A | Yes | N/A | N/A |

---

## Next Steps

- **[Stability Prediction Tutorial](../tutorials/stability-prediction.md)** --- fine-tune for predicting protein stability
- **[LoRA for Small Datasets](../tutorials/lora-small-datasets.md)** --- techniques for limited data
- **[Experiment Tracking](../tutorials/experiment-tracking.md)** --- compare runs with WandB, Comet, or MLflow
- **[Training Strategies Reference](../reference/training/strategies.md)** --- full parameter documentation
