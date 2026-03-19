# Attention Modules

Pluggable attention implementations registered in the `ATTENTION_REGISTRY`.
All implementations subclass `BaseAttention` and share a common forward
signature.

## Quick Start

```python
from molfun.modules.attention import ATTENTION_REGISTRY

# List available attention mechanisms
print(ATTENTION_REGISTRY.list())
# ["flash", "gated", "linear", "standard"]

# Build an attention module
attn = ATTENTION_REGISTRY.build("flash", num_heads=8, head_dim=64)

# Use in a model via swap
from molfun import MolfunStructureModel
model = MolfunStructureModel.from_pretrained("openfold_v2")
model.swap("attention", "flash")
```

## ATTENTION_REGISTRY

::: molfun.modules.attention.ATTENTION_REGISTRY
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

---

## BaseAttention

::: molfun.modules.attention.base.BaseAttention
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Abstract base class for all attention modules.

```python
from molfun.modules.attention.base import BaseAttention

class MyAttention(BaseAttention):
    def forward(self, q, k, v, mask=None, bias=None):
        ...

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim
```

### Forward Signature

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | `Tensor` | Query tensor `(B, H, L, D)` |
| `k` | `Tensor` | Key tensor `(B, H, L, D)` |
| `v` | `Tensor` | Value tensor `(B, H, L, D)` |
| `mask` | `Tensor \| None` | Attention mask `(B, 1, L, L)` or `(B, H, L, L)` |
| `bias` | `Tensor \| None` | Attention bias (e.g., pair representation) |

**Returns:** `Tensor` of shape `(B, H, L, D)`.

---

## FlashAttention

::: molfun.modules.attention.flash.FlashAttention
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

GPU-optimized attention using Flash Attention 2. Requires a CUDA device
and `flash-attn` package.

```python
from molfun.modules.attention import ATTENTION_REGISTRY

attn = ATTENTION_REGISTRY.build("flash", num_heads=8, head_dim=64)
output = attn(q, k, v, mask=mask)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_heads` | `int` | *required* | Number of attention heads |
| `head_dim` | `int` | *required* | Dimension per head |
| `dropout` | `float` | `0.0` | Attention dropout (training only) |

---

## StandardAttention

::: molfun.modules.attention.standard.StandardAttention
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Standard scaled dot-product attention. Works on all devices.

```python
attn = ATTENTION_REGISTRY.build("standard", num_heads=8, head_dim=64)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_heads` | `int` | *required* | Number of attention heads |
| `head_dim` | `int` | *required* | Dimension per head |
| `dropout` | `float` | `0.0` | Attention dropout |

---

## LinearAttention

::: molfun.modules.attention.linear.LinearAttention
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Linear-complexity attention using kernel feature maps. Suitable for very
long sequences.

```python
attn = ATTENTION_REGISTRY.build("linear", num_heads=8, head_dim=64)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_heads` | `int` | *required* | Number of attention heads |
| `head_dim` | `int` | *required* | Dimension per head |
| `feature_map` | `str` | `"elu"` | Kernel feature map (`"elu"`, `"relu"`, `"favor+"`) |

---

## GatedAttention

::: molfun.modules.attention.gated.GatedAttention
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Attention with gating mechanism as used in AlphaFold2 / OpenFold.

```python
attn = ATTENTION_REGISTRY.build("gated", num_heads=8, head_dim=64)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_heads` | `int` | *required* | Number of attention heads |
| `head_dim` | `int` | *required* | Dimension per head |
| `dropout` | `float` | `0.0` | Attention dropout |

The output is element-wise multiplied by a learned gating vector before
the final linear projection.
