---
title: Stability Prediction with HeadOnly
---

# Stability Prediction with HeadOnly

<span class="badge badge-beginner">Beginner</span> &nbsp; ~20 min

Predict protein thermostability (DDG values) from amino acid sequences using a **HeadOnly**
fine-tuning strategy. This is the simplest way to adapt a pretrained structure model to a
new regression task.

---

## What You Will Learn

- Load a CSV dataset of protein sequences with stability labels
- Create a `StructureDataset` and DataLoader
- Fine-tune with `HeadOnlyFinetune` (only the prediction head is trained)
- Evaluate predictions with a scatter plot

## Prerequisites

- Molfun installed (`pip install molfun`)
- A CSV file with columns `sequence` and `ddg` (or similar stability metric)

---

## Step 1: Prepare Your Data

For this tutorial we will use a CSV of protein sequences annotated with experimentally
measured DDG (change in Gibbs free energy upon mutation) values. You can use any CSV
with a `sequence` column and a numeric label column.

```python
import pandas as pd

# Load your stability dataset
df = pd.read_csv("stability_data.csv")
print(df.head())
```

```
          sequence                                              ddg
0  MKFLILLFNILCLFPVLAADNHGVS...                              -1.2
1  MVLSPADKTNVKAAWGKVGAHAGEYGAE...                            0.8
2  MNIFEMLRIDEGLRLKIYKDTEGYYTIG...                           -2.5
...
```

!!! info "Example datasets"

    If you do not have a stability dataset, the [ProThermDB](https://web.iitm.ac.in/bioinfo2/prothermdb/)
    and [FireProtDB](https://loschmidt.chemi.muni.cz/fireprotdb/) databases are good
    public sources of protein stability measurements.

---

## Step 2: Create a StructureDataset

Molfun's `StructureDataset` wraps your sequences and labels into a format the model
expects.

```python
from molfun.data import StructureDataset, DataSplitter
from torch.utils.data import DataLoader

# Create dataset from sequences and labels
dataset = StructureDataset(
    sequences=df["sequence"].tolist(),
    labels=df["ddg"].values,       # NumPy array or list of floats
    max_length=512,                # Truncate sequences longer than 512 residues
)

# Split into train / validation (80/20)
splitter = DataSplitter(test_size=0.2, random_state=42)
train_dataset, val_dataset = splitter.split(dataset)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
```

---

## Step 3: Load a Pretrained Model

Load a pretrained OpenFold model and attach an **affinity head** --- a lightweight MLP
that maps the trunk's structural embeddings to a scalar prediction.

```python
from molfun import MolfunStructureModel

model = MolfunStructureModel.from_pretrained(
    "openfold_v1",                # Pretrained model name
    device="cuda",                # Use "cpu" if no GPU
    head="affinity",              # Attach a regression head
    head_config={
        "hidden_dim": 256,        # Hidden layer size
        "num_layers": 2,          # Number of MLP layers
        "dropout": 0.1,
    },
)
```

!!! note "Why `head='affinity'`?"

    The `"affinity"` head is a general-purpose regression head that outputs a single
    scalar per input. It works for any sequence-to-value task: stability, binding
    affinity, solubility, etc. The name reflects its most common use case.

---

## Step 4: Fine-Tune with HeadOnly

The `HeadOnlyFinetune` strategy freezes the entire pretrained trunk and only trains the
prediction head. This is fast, avoids overfitting on small datasets, and preserves the
learned structural representations.

```python
from molfun.training import HeadOnlyFinetune

strategy = HeadOnlyFinetune(
    lr=1e-3,               # Learning rate for the head
    weight_decay=1e-4,     # L2 regularization
)

model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    strategy=strategy,
    epochs=20,
)
```

??? example "Expected training output"

    ```
    Epoch  1/20 | Train Loss: 4.213 | Val Loss: 3.891
    Epoch  2/20 | Train Loss: 3.102 | Val Loss: 2.876
    Epoch  3/20 | Train Loss: 2.341 | Val Loss: 2.198
    ...
    Epoch 20/20 | Train Loss: 0.412 | Val Loss: 0.523
    ```

---

## Step 5: Evaluate with a Scatter Plot

Run inference on the validation set and compare predicted vs. actual DDG values.

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

model.eval()
predictions, actuals = [], []

with torch.no_grad():
    for batch in val_loader:
        output = model.predict(batch["sequence"])
        predictions.extend(output.scores.cpu().numpy())
        actuals.extend(batch["labels"].cpu().numpy())

predictions = np.array(predictions)
actuals = np.array(actuals)

# Compute Pearson correlation
from scipy.stats import pearsonr
r, p_value = pearsonr(actuals, predictions)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(actuals, predictions, alpha=0.6, s=20)
ax.plot(
    [actuals.min(), actuals.max()],
    [actuals.min(), actuals.max()],
    "r--", linewidth=1.5,
)
ax.set_xlabel("Actual DDG (kcal/mol)")
ax.set_ylabel("Predicted DDG (kcal/mol)")
ax.set_title(f"Stability Prediction (r = {r:.3f})")
plt.tight_layout()
plt.savefig("stability_scatter.png", dpi=150)
plt.show()
```

---

## Step 6: Save the Model

```python
# Save locally
model.save("models/stability_headonly")

# Or push to HuggingFace Hub
model.push_to_hub("your-username/stability-predictor")
```

---

## Full Script

??? abstract "Complete runnable code"

    ```python
    """Stability prediction with HeadOnly fine-tuning."""
    import pandas as pd
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    from torch.utils.data import DataLoader

    from molfun import MolfunStructureModel
    from molfun.data import StructureDataset, DataSplitter
    from molfun.training import HeadOnlyFinetune

    # ── Data ──────────────────────────────────────────────
    df = pd.read_csv("stability_data.csv")

    dataset = StructureDataset(
        sequences=df["sequence"].tolist(),
        labels=df["ddg"].values,
        max_length=512,
    )

    splitter = DataSplitter(test_size=0.2, random_state=42)
    train_dataset, val_dataset = splitter.split(dataset)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # ── Model ─────────────────────────────────────────────
    model = MolfunStructureModel.from_pretrained(
        "openfold_v1",
        device="cuda",
        head="affinity",
        head_config={"hidden_dim": 256, "num_layers": 2, "dropout": 0.1},
    )

    # ── Train ─────────────────────────────────────────────
    strategy = HeadOnlyFinetune(lr=1e-3, weight_decay=1e-4)
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        strategy=strategy,
        epochs=20,
    )

    # ── Evaluate ──────────────────────────────────────────
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in val_loader:
            output = model.predict(batch["sequence"])
            predictions.extend(output.scores.cpu().numpy())
            actuals.extend(batch["labels"].cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    r, _ = pearsonr(actuals, predictions)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actuals, predictions, alpha=0.6, s=20)
    ax.plot(
        [actuals.min(), actuals.max()],
        [actuals.min(), actuals.max()],
        "r--", linewidth=1.5,
    )
    ax.set_xlabel("Actual DDG (kcal/mol)")
    ax.set_ylabel("Predicted DDG (kcal/mol)")
    ax.set_title(f"Stability Prediction (r = {r:.3f})")
    plt.tight_layout()
    plt.savefig("stability_scatter.png", dpi=150)

    # ── Save ──────────────────────────────────────────────
    model.save("models/stability_headonly")
    ```

---

## Next Steps

- **Want better accuracy?** Try [LoRA for Small Datasets](lora-small-datasets.md) to
  also update the trunk with parameter-efficient fine-tuning.
- **Have binding affinity data?** See [Binding Affinity Prediction](binding-affinity.md).
- **Want to track experiments?** Add a tracker in one line --- see
  [Experiment Tracking](experiment-tracking.md).
