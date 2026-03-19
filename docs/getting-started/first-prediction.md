---
title: First Prediction
---

# Your First Prediction

This tutorial walks through a complete prediction workflow: loading a pretrained model,
running inference on a protein sequence, exploring the output, saving a PDB file, and
visualizing the 3D structure.

!!! info "Time estimate: 10 minutes"

    You will need Molfun installed (`pip install molfun`). A GPU is not required ---
    all examples run on CPU.

---

## Step 1: Load a Pretrained Model

`MolfunStructureModel.from_pretrained()` downloads and initializes a pretrained backend.
Currently, the `openfold` backend is available, with ESMFold and Protenix coming soon.

```python
from molfun import MolfunStructureModel

model = MolfunStructureModel.from_pretrained(
    name="openfold",   # (1)!
    device="cpu",       # (2)!
)
```

1. The backend name. Use `"openfold"` for the AlphaFold2-based architecture.
2. Set to `"cuda"` if you have a GPU. The model and inputs will be moved automatically.

!!! tip "Custom heads"

    You can attach a custom prediction head at load time:

    ```python
    model = MolfunStructureModel.from_pretrained(
        name="openfold",
        head="affinity",
        head_config={"hidden_dim": 256, "output_dim": 1},
    )
    ```

    This replaces the default structure module output with a task-specific head.

---

## Step 2: Run Prediction

Pass a protein sequence to `model.predict()`. The sequence should use standard one-letter
amino acid codes.

```python
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL"

output = model.predict(sequence)  # (1)!
```

1. You can also pass a `Batch` object for batched inference or to include MSA features.

The call returns a `TrunkOutput` object containing the model's predictions.

---

## Step 3: Explore the Output

`TrunkOutput` exposes four key attributes:

```python
# Per-residue single representation — shape: (N_residues, d_single)
print(f"Single repr: {output.single_repr.shape}")

# Pairwise representation — shape: (N_residues, N_residues, d_pair)
print(f"Pair repr:   {output.pair_repr.shape}")

# Predicted 3D coordinates — shape: (N_residues, 37, 3)
# 37 corresponds to the maximum number of atoms per residue
print(f"Coordinates: {output.structure_coords.shape}")

# Per-residue confidence (pLDDT) — shape: (N_residues,)
print(f"Confidence:  {output.confidence.shape}")
print(f"Mean pLDDT:  {output.confidence.mean():.1f}")
```

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `single_repr` | `(N, d_single)` | Per-residue embeddings from the trunk. Useful for downstream tasks like property prediction. |
| `pair_repr` | `(N, N, d_pair)` | Pairwise residue embeddings. Encodes distance and orientation information. |
| `structure_coords` | `(N, 37, 3)` | Predicted atom coordinates in angstroms. Dimension 37 covers all standard atom types. |
| `confidence` | `(N,)` | Per-residue pLDDT scores (0--100). Higher means more confident. |

!!! info "When to use representations vs. coordinates"

    - Use **`structure_coords`** when you need the 3D structure itself (visualization,
      RMSD computation, contact maps).
    - Use **`single_repr`** and **`pair_repr`** as features for downstream models
      (binding affinity, stability, function prediction).

---

## Step 4: Save as PDB

For a PDB file you can write directly to disk, use the `predict_structure` convenience
function which returns a ready-made PDB string:

```python
from molfun import predict_structure

result = predict_structure(
    sequence=sequence,
    backend="openfold",
    device="cpu",
)

# Write the PDB file
with open("my_protein.pdb", "w") as f:
    f.write(result["pdb_string"])

print("Saved my_protein.pdb")
```

The returned dictionary contains:

```python
result["coordinates"]  # numpy array, shape (N, 37, 3)
result["plddt"]        # numpy array, shape (N,)
result["pdb_string"]   # full PDB-format string, ready to write
```

---

## Step 5: Visualize the Structure

### Option A: py3Dmol (Jupyter Notebook)

If you are working in a Jupyter notebook, `py3Dmol` renders interactive 3D structures
inline:

```python
import py3Dmol

# Read the PDB we just saved
with open("my_protein.pdb") as f:
    pdb_data = f.read()

view = py3Dmol.view(width=800, height=600)
view.addModel(pdb_data, "pdb")

# Color by pLDDT (B-factor column)
view.setStyle({"cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb", "min": 50, "max": 90}}})
view.zoomTo()
view.show()
```

!!! tip "Color scale"

    The pLDDT scores are stored in the B-factor column of the PDB file. The `roygb`
    gradient maps low confidence (red) to high confidence (blue), matching the AlphaFold
    coloring convention.

### Option B: PyMOL (Desktop)

Open the PDB in PyMOL from the command line:

```bash
pymol my_protein.pdb
```

Then in the PyMOL console, color by B-factor (pLDDT):

```
spectrum b, red_white_blue, minimum=50, maximum=90
```

### Option C: Programmatic Analysis

Use Biopython to parse the structure for further analysis:

```python
from Bio.PDB import PDBParser

parser = PDBParser(QUIET=True)
structure = parser.get_structure("pred", "my_protein.pdb")

for model in structure:
    for chain in model:
        for residue in chain:
            ca = residue["CA"]
            print(f"{residue.get_resname()} {residue.get_id()[1]}: "
                  f"CA at ({ca.get_vector().get_array()})")
```

---

## Full Example

Here is the complete workflow in one script:

```python
from molfun import MolfunStructureModel, predict_structure

# --- Model-level API (full control) ---
model = MolfunStructureModel.from_pretrained("openfold", device="cpu")

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL"
output = model.predict(sequence)

print(f"Structure coords: {output.structure_coords.shape}")
print(f"Mean pLDDT:       {output.confidence.mean():.1f}")
print(f"Single repr dim:  {output.single_repr.shape[-1]}")

# --- Convenience API (PDB output) ---
result = predict_structure(sequence=sequence, backend="openfold", device="cpu")

with open("my_protein.pdb", "w") as f:
    f.write(result["pdb_string"])

print("Saved my_protein.pdb")
```

---

## Next Steps

Now that you can make predictions, learn how to improve them with your own data:

- **[First Fine-Tuning](first-finetuning.md)** --- LoRA fine-tuning end to end
- **[Binding Affinity Tutorial](../tutorials/binding-affinity.md)** --- predict protein-ligand interactions
- **[API Reference: predict functions](../reference/api/predict.md)** --- full parameter docs
