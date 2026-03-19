# Trunk Blocks

Trunk blocks form the core of the model architecture. Each block processes
single and pair representations through attention and feed-forward layers.
All implementations subclass `BaseBlock` and are registered in
`BLOCK_REGISTRY`.

## Quick Start

```python
from molfun.modules.blocks import BLOCK_REGISTRY

# List available blocks
print(BLOCK_REGISTRY.list())
# ["evoformer", "pairformer", "simple_transformer"]

# Build a block
block = BLOCK_REGISTRY.build("pairformer", c_s=256, c_z=128, num_heads=8)

# Swap blocks in a live model
from molfun import MolfunStructureModel
model = MolfunStructureModel.from_pretrained("openfold_v2")
model.swap_all("block", "pairformer")
```

## BLOCK_REGISTRY

::: molfun.modules.blocks.BLOCK_REGISTRY
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

---

## BaseBlock

::: molfun.modules.blocks.base.BaseBlock
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Abstract base class for trunk blocks.

### Forward Signature

| Parameter | Type | Description |
|-----------|------|-------------|
| `s` | `Tensor` | Single representation `(B, L, c_s)` |
| `z` | `Tensor` | Pair representation `(B, L, L, c_z)` |
| `mask` | `Tensor \| None` | Sequence mask `(B, L)` |

**Returns:** `BlockOutput` with `.s` (single) and `.z` (pair) tensors.

---

## BlockOutput

::: molfun.modules.blocks.base.BlockOutput
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Dataclass holding block output tensors.

| Field | Type | Description |
|-------|------|-------------|
| `s` | `Tensor` | Updated single representation |
| `z` | `Tensor` | Updated pair representation |

---

## Pairformer

::: molfun.modules.blocks.pairformer.PairformerBlock
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Pairformer block with triangular multiplicative updates and attention.

```python
block = BLOCK_REGISTRY.build(
    "pairformer",
    c_s=256,
    c_z=128,
    num_heads=8,
    dropout=0.0,
)
output = block(s, z, mask=mask)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_s` | `int` | *required* | Single representation dimension |
| `c_z` | `int` | *required* | Pair representation dimension |
| `num_heads` | `int` | `8` | Number of attention heads |
| `dropout` | `float` | `0.0` | Dropout probability |

---

## Evoformer

::: molfun.modules.blocks.evoformer.EvoformerBlock
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

The Evoformer block from AlphaFold2, with MSA row/column attention and
outer product mean for pair updates.

```python
block = BLOCK_REGISTRY.build(
    "evoformer",
    c_s=256,
    c_z=128,
    c_m=256,
    num_heads=8,
)
output = block(s, z, mask=mask)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_s` | `int` | *required* | Single representation dimension |
| `c_z` | `int` | *required* | Pair representation dimension |
| `c_m` | `int` | `256` | MSA representation dimension |
| `num_heads` | `int` | `8` | Number of attention heads |
| `dropout` | `float` | `0.0` | Dropout probability |

---

## SimpleTransformer

::: molfun.modules.blocks.simple_transformer.SimpleTransformerBlock
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

A lightweight transformer block with standard self-attention and FFN.
Useful for baselines and property prediction heads.

```python
block = BLOCK_REGISTRY.build(
    "simple_transformer",
    c_s=256,
    num_heads=8,
    ffn_dim=1024,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_s` | `int` | *required* | Input/output dimension |
| `num_heads` | `int` | `8` | Number of attention heads |
| `ffn_dim` | `int \| None` | `None` | FFN hidden dimension (defaults to `4 * c_s`) |
| `dropout` | `float` | `0.0` | Dropout probability |
