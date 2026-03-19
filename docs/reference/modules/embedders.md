# Embedders

Embedders convert raw amino acid sequences (and optional features like MSAs)
into the initial single and pair representations consumed by trunk blocks.
All implementations subclass `BaseEmbedder` and are registered in
`EMBEDDER_REGISTRY`.

## Quick Start

```python
from molfun.modules.embedders import EMBEDDER_REGISTRY

# List available embedders
print(EMBEDDER_REGISTRY.list())
# ["esm", "input"]

# Build an embedder
embedder = EMBEDDER_REGISTRY.build("input", c_s=256, c_z=128)
output = embedder(aatype, residue_index)

# Swap embedder in a live model
from molfun import MolfunStructureModel
model = MolfunStructureModel.from_pretrained("openfold_v2")
model.swap("embedder", "esm")
```

## EMBEDDER_REGISTRY

::: molfun.modules.embedders.EMBEDDER_REGISTRY
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

---

## BaseEmbedder

::: molfun.modules.embedders.base.BaseEmbedder
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Abstract base class for all embedders.

### Forward Signature

| Parameter | Type | Description |
|-----------|------|-------------|
| `aatype` | `Tensor` | Amino acid type indices `(B, L)` |
| `residue_index` | `Tensor` | Residue position indices `(B, L)` |
| `**kwargs` | `dict` | Additional features (e.g., MSA, templates) |

**Returns:** `EmbedderOutput`

---

## EmbedderOutput

::: molfun.modules.embedders.base.EmbedderOutput
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Dataclass holding embedder outputs.

| Field | Type | Description |
|-------|------|-------------|
| `s` | `Tensor` | Single representation `(B, L, c_s)` |
| `z` | `Tensor` | Pair representation `(B, L, L, c_z)` |

---

## InputEmbedder

::: molfun.modules.embedders.input_embedder.InputEmbedder
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Standard input embedder that computes single and pair representations
from amino acid types and relative positional encodings.

```python
embedder = EMBEDDER_REGISTRY.build(
    "input",
    c_s=256,
    c_z=128,
    max_relative_position=32,
)

output = embedder(aatype, residue_index)
s = output.s   # (B, L, 256)
z = output.z   # (B, L, L, 128)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_s` | `int` | *required* | Single representation dimension |
| `c_z` | `int` | *required* | Pair representation dimension |
| `max_relative_position` | `int` | `32` | Maximum relative positional encoding distance |
| `num_amino_acids` | `int` | `21` | Number of amino acid types (including unknown) |

---

## ESMEmbedder

::: molfun.modules.embedders.esm_embedder.ESMEmbedder
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Embedder that uses ESM (Evolutionary Scale Modeling) language model
representations as initial features. Provides richer single representations
from pretrained protein language models.

```python
embedder = EMBEDDER_REGISTRY.build(
    "esm",
    c_s=256,
    c_z=128,
    esm_model="esm2_t33_650M_UR50D",
    freeze_esm=True,
)

output = embedder(aatype, residue_index)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_s` | `int` | *required* | Single representation dimension |
| `c_z` | `int` | *required* | Pair representation dimension |
| `esm_model` | `str` | `"esm2_t33_650M_UR50D"` | ESM model name |
| `freeze_esm` | `bool` | `True` | Whether to freeze ESM weights |
| `layer_idx` | `int` | `-1` | Which ESM layer to extract representations from |

!!! note
    The ESM embedder requires the `fair-esm` package. Install with
    `pip install fair-esm`.
