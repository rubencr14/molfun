---
title: Quick Start
---

# Quick Start

Three examples to see what Molfun can do --- all in under five minutes.

---

## 1. Predict a Protein Structure

Given an amino acid sequence, predict its 3D coordinates and per-residue confidence scores.

=== "Python"

    ```python
    from molfun import predict_structure

    result = predict_structure(
        sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL",
        backend="openfold",
        device="cpu",
    )

    print(f"Coordinates shape: {result['coordinates'].shape}")  # (N_residues, 37, 3)
    print(f"Mean pLDDT: {result['plddt'].mean():.1f}")

    # Save directly to a PDB file
    with open("prediction.pdb", "w") as f:
        f.write(result["pdb_string"])
    ```

=== "CLI"

    ```bash
    molfun structure \
        --sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL" \
        --output prediction.pdb
    ```

!!! tip "One function, one line"

    `predict_structure` is a convenience wrapper around `MolfunStructureModel`. For more
    control --- custom heads, batched inference, or access to intermediate representations ---
    see [First Prediction](first-prediction.md).

---

## 2. Fine-Tune with LoRA

Fine-tune a pretrained model on your own data using LoRA --- only a handful of lines.

=== "Python"

    ```python
    from molfun import MolfunStructureModel
    from molfun.training import LoRAFinetune
    from torch.utils.data import DataLoader

    # 1. Load a pretrained model
    model = MolfunStructureModel.from_pretrained("openfold", device="cpu")

    # 2. Prepare your data (any PyTorch Dataset / DataLoader)
    train_loader = DataLoader(your_dataset, batch_size=2, shuffle=True)

    # 3. Fine-tune with LoRA
    strategy = LoRAFinetune(rank=8, alpha=16.0, lr_lora=1e-4, lr_head=1e-3)

    metrics = model.fit(
        train_loader=train_loader,
        strategy=strategy,
        epochs=10,
        tracker="wandb",           # optional: track with Weights & Biases
        checkpoint_dir="./ckpts",  # optional: save checkpoints
    )

    # 4. Save the fine-tuned model
    model.save("my_finetuned_model")
    ```

=== "CLI"

    ```bash
    molfun fit \
        --data ./my_dataset \
        --strategy lora \
        --rank 8 \
        --epochs 10 \
        --tracker wandb \
        --output ./my_finetuned_model
    ```

!!! info "Only ~0.5% of parameters are trained"

    With rank 8, LoRA adds low-rank adapters to the attention layers while keeping the
    full trunk frozen. This makes fine-tuning feasible on a single GPU with limited data.
    See [First Fine-Tuning](first-finetuning.md) for a detailed walkthrough.

---

## 3. Predict Binding Affinity

Estimate how strongly a protein binds to a small-molecule ligand.

=== "Python"

    ```python
    from molfun import predict_affinity

    result = predict_affinity(
        sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL",
        ligand_smiles="CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        device="cpu",
    )

    print(f"Binding affinity: {result['binding_affinity_kcal']:.2f} kcal/mol")
    print(f"Confidence: {result['confidence']:.2f}")
    ```

=== "CLI"

    ```bash
    molfun affinity \
        --sequence "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL" \
        --ligand "CC(=O)Oc1ccccc1C(=O)O" \
        --device cpu
    ```

---

## Bonus: Molecular Properties

Compute physicochemical properties from sequence alone --- no model weights needed.

```python
from molfun import predict_properties

props = predict_properties(
    sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL",
    properties=["molecular_weight", "isoelectric_point", "hydrophobicity", "charge"],
)

for name, value in props.items():
    print(f"{name}: {value:.2f}")
```

---

## What is Next?

| Goal | Guide |
|------|-------|
| Understand each prediction step in detail | [First Prediction](first-prediction.md) |
| Run a full fine-tuning workflow with evaluation | [First Fine-Tuning](first-finetuning.md) |
| Explore real-world use cases | [Tutorials](../tutorials/index.md) |
| Dive into the API reference | [MolfunStructureModel](../reference/api/model.md) |
