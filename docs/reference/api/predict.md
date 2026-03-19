# Predict Functions

High-level prediction functions that handle model loading, caching, and
inference in a single call. These are the simplest entry points for running
predictions without manually managing model instances.

## Quick Start

```python
from molfun import predict_structure, predict_properties, predict_affinity

# Structure prediction
result = predict_structure("MKFLILLFNILCLFPVLAADNH...")

# Property prediction
props = predict_properties("MKFLILLFNILCLFPVLAADNH...", properties=["plddt", "disorder"])

# Binding affinity
affinity = predict_affinity(
    protein_seq="MKFLILLFNILCLFPVLAADNH...",
    ligand_sdf="ligand.sdf",
)
```

## Functions

### predict_structure

::: molfun.predict.predict_structure
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

Predict the 3D structure of a protein from its amino acid sequence.

```python
from molfun import predict_structure

result = predict_structure(
    sequence="MKFLILLFNILCLFPVLAADNH...",
    model="openfold_v2",
    num_recycles=3,
    device="cuda",
)

# Access outputs
coords = result.positions   # (N_residues, 37, 3) atom coordinates
plddt  = result.plddt       # (N_residues,) per-residue confidence
pae    = result.pae         # (N_residues, N_residues) predicted aligned error
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequence` | `str` | *required* | Amino acid sequence (one-letter codes) |
| `model` | `str` | `"openfold_v2"` | Pretrained model name |
| `num_recycles` | `int` | `3` | Number of recycling iterations |
| `msa` | `str \| None` | `None` | Path to A3M MSA file |
| `device` | `str` | `"cpu"` | Compute device |
| `dtype` | `torch.dtype` | `torch.float32` | Model precision |

**Returns:** `TrunkOutput` with `.positions`, `.plddt`, `.pae`.

---

### predict_properties

::: molfun.predict.predict_properties
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

Predict per-residue or global protein properties.

```python
from molfun import predict_properties

props = predict_properties(
    sequence="MKFLILLFNILCLFPVLAADNH...",
    properties=["plddt", "disorder", "secondary_structure"],
    model="openfold_v2",
)

plddt = props["plddt"]          # (N_residues,)
ss    = props["secondary_structure"]  # (N_residues,) H/E/C labels
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequence` | `str` | *required* | Amino acid sequence |
| `properties` | `list[str]` | `["plddt"]` | Properties to predict |
| `model` | `str` | `"openfold_v2"` | Pretrained model name |
| `device` | `str` | `"cpu"` | Compute device |

**Returns:** `dict[str, Tensor]` mapping property names to tensors.

---

### predict_affinity

::: molfun.predict.predict_affinity
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

Predict binding affinity between a protein and a ligand.

```python
from molfun import predict_affinity

result = predict_affinity(
    protein_seq="MKFLILLFNILCLFPVLAADNH...",
    ligand_sdf="ligand.sdf",
    model="openfold_v2",
)

print(f"Predicted pKd: {result.pkd:.2f}")
print(f"Confidence: {result.confidence:.2f}")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `protein_seq` | `str` | *required* | Protein amino acid sequence |
| `ligand_sdf` | `str` | *required* | Path to ligand SDF file |
| `model` | `str` | `"openfold_v2"` | Pretrained model name |
| `device` | `str` | `"cpu"` | Compute device |

**Returns:** Result object with `.pkd` (predicted pKd) and `.confidence`.

---

### clear_cache

::: molfun.predict.clear_cache
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

Clear the internal model cache used by the predict functions.

```python
from molfun.predict import clear_cache

# Free memory by releasing cached models
clear_cache()
```

This is useful when switching between different models or when GPU memory
is limited. The next call to any predict function will reload the model
from disk.
