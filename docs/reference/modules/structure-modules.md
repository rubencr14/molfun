# Structure Modules

Structure modules convert trunk representations into 3D atomic coordinates.
They operate on the single representation and pair representation to produce
per-residue frames and atom positions.

## Quick Start

```python
from molfun.modules.structure_module import STRUCTURE_MODULE_REGISTRY

# List available structure modules
print(STRUCTURE_MODULE_REGISTRY.list())
# ["diffusion", "ipa"]

# Build a structure module
sm = STRUCTURE_MODULE_REGISTRY.build("ipa", c_s=384, c_z=128, num_heads=12)

# Swap in a model
from molfun import MolfunStructureModel
model = MolfunStructureModel.from_pretrained("openfold_v2")
model.swap("structure_module", "diffusion")
```

## STRUCTURE_MODULE_REGISTRY

::: molfun.modules.structure_module.STRUCTURE_MODULE_REGISTRY
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

---

## BaseStructureModule

::: molfun.modules.structure_module.base.BaseStructureModule
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Abstract base class for structure modules.

### Forward Signature

| Parameter | Type | Description |
|-----------|------|-------------|
| `s` | `Tensor` | Single representation `(B, L, c_s)` |
| `z` | `Tensor` | Pair representation `(B, L, L, c_z)` |
| `mask` | `Tensor \| None` | Sequence mask `(B, L)` |

**Returns:** `StructureModuleOutput`

---

## StructureModuleOutput

::: molfun.modules.structure_module.base.StructureModuleOutput
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Dataclass holding structure module outputs.

| Field | Type | Description |
|-------|------|-------------|
| `positions` | `Tensor` | Atom positions `(B, L, 37, 3)` in Angstroms |
| `frames` | `Tensor` | Backbone frames `(B, L, 4, 4)` as rigid transforms |
| `plddt` | `Tensor` | Per-residue confidence `(B, L)` in [0, 1] |
| `pae` | `Tensor \| None` | Predicted aligned error `(B, L, L)` |

---

## IPA (Invariant Point Attention)

::: molfun.modules.structure_module.ipa.IPAStructureModule
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Invariant Point Attention structure module from AlphaFold2. Iteratively
refines backbone frames using geometric attention that is invariant to
global rotations and translations.

```python
sm = STRUCTURE_MODULE_REGISTRY.build(
    "ipa",
    c_s=384,
    c_z=128,
    num_heads=12,
    num_layers=8,
    num_query_points=4,
    num_value_points=8,
)

output = sm(s, z, mask=mask)
coords = output.positions    # (B, L, 37, 3)
plddt  = output.plddt        # (B, L)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_s` | `int` | *required* | Single representation dimension |
| `c_z` | `int` | *required* | Pair representation dimension |
| `num_heads` | `int` | `12` | Number of IPA heads |
| `num_layers` | `int` | `8` | Number of IPA refinement layers |
| `num_query_points` | `int` | `4` | Number of query points per head |
| `num_value_points` | `int` | `8` | Number of value points per head |

---

## Diffusion

::: molfun.modules.structure_module.diffusion.DiffusionStructureModule
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Diffusion-based structure module that generates 3D coordinates through
iterative denoising, inspired by diffusion models for molecular generation.

```python
sm = STRUCTURE_MODULE_REGISTRY.build(
    "diffusion",
    c_s=384,
    c_z=128,
    num_steps=100,
    noise_schedule="cosine",
)

output = sm(s, z, mask=mask)
coords = output.positions
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_s` | `int` | *required* | Single representation dimension |
| `c_z` | `int` | *required* | Pair representation dimension |
| `num_steps` | `int` | `100` | Number of diffusion denoising steps |
| `noise_schedule` | `str` | `"cosine"` | Noise schedule (`"cosine"`, `"linear"`) |
| `num_heads` | `int` | `8` | Number of attention heads in the denoiser |
