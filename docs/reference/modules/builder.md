# ModelBuilder

Programmatic assembly of custom model architectures from registered
modular components.

## Quick Start

```python
from molfun.modules.builder import ModelBuilder

# Build a custom model
builder = ModelBuilder(
    embedder="input",
    block="pairformer",
    n_blocks=48,
    structure_module="ipa",
    configs={
        "c_s": 256,
        "c_z": 128,
        "num_heads": 8,
    },
)

model = builder.build()
print(model)
```

## Class Reference

::: molfun.modules.builder.ModelBuilder
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### Constructor

```python
builder = ModelBuilder(
    embedder="input",
    block="pairformer",
    n_blocks=48,
    structure_module="ipa",
    configs={
        "c_s": 384,
        "c_z": 128,
        "num_heads": 8,
        "head_dim": 32,
    },
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedder` | `str` | *required* | Registered embedder name (from `EMBEDDER_REGISTRY`) |
| `block` | `str` | *required* | Registered block name (from `BLOCK_REGISTRY`) |
| `n_blocks` | `int` | *required* | Number of trunk blocks to stack |
| `structure_module` | `str` | *required* | Registered structure module name (from `STRUCTURE_MODULE_REGISTRY`) |
| `configs` | `dict` | `{}` | Shared configuration passed to each component |

---

### build

Assemble and return the complete model.

```python
model = builder.build()
```

**Returns:** `BuiltModel` -- an `nn.Module` containing the embedder, trunk
(stacked blocks), and structure module.

---

## BuiltModel

The `BuiltModel` returned by `builder.build()` is a standard `nn.Module`
with the following structure:

```
BuiltModel
  .embedder          -> BaseEmbedder
  .trunk             -> nn.ModuleList[BaseBlock]
  .structure_module  -> BaseStructureModule
```

### Forward Pass

```python
output = model(aatype, residue_index, mask=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `aatype` | `Tensor` | Amino acid types `(B, L)` |
| `residue_index` | `Tensor` | Residue indices `(B, L)` |
| `mask` | `Tensor \| None` | Sequence mask `(B, L)` |

**Returns:** `StructureModuleOutput`

---

## Example: Custom Architecture

```python
from molfun.modules.builder import ModelBuilder
from molfun import MolfunStructureModel

# Define a lightweight model for fast iteration
builder = ModelBuilder(
    embedder="input",
    block="simple_transformer",
    n_blocks=12,
    structure_module="ipa",
    configs={
        "c_s": 128,
        "c_z": 64,
        "num_heads": 4,
    },
)

built = builder.build()

# Wrap in MolfunStructureModel for full API access
model = MolfunStructureModel.from_custom(
    embedder="input",
    block="simple_transformer",
    n_blocks=12,
    structure_module="ipa",
    c_s=128,
    c_z=64,
    num_heads=4,
)

model.fit(train_dataset=ds, strategy="full", epochs=20)
```
