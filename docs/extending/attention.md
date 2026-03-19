# Adding Attention Modules

This guide walks through adding a custom attention mechanism to Molfun. Attention modules are the lowest-level pluggable component -- they are used inside trunk blocks (Evoformer, Pairformer, etc.) and can be swapped independently of the block architecture.

## The BaseAttention interface

All attention modules inherit from `BaseAttention` in `molfun/modules/attention/base.py`:

```python
class BaseAttention(ABC, nn.Module):
    """Maps (Q, K, V) -> attended output."""

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,       # [B, H, Lq, D]
        k: torch.Tensor,       # [B, H, Lk, D]
        v: torch.Tensor,       # [B, H, Lk, D]
        mask: Optional[torch.Tensor] = None,   # [B, 1|H, Lq, Lk]
        bias: Optional[torch.Tensor] = None,   # [B, 1|H, Lq, Lk]
    ) -> torch.Tensor:  # [B, H, Lq, D]
        ...

    @property
    @abstractmethod
    def num_heads(self) -> int: ...

    @property
    @abstractmethod
    def head_dim(self) -> int: ...

    @property
    def embed_dim(self) -> int:
        return self.num_heads * self.head_dim
```

Key design points:

- Input tensors already have the head dimension split out -- no reshaping needed inside `forward`.
- `mask` is a boolean tensor where `True` means "attend" and `False` means "ignore."
- `bias` is an additive bias applied to attention logits before softmax (used by Evoformer pair bias).
- The `embed_dim` property is derived automatically from `num_heads * head_dim`.

### AttentionConfig

A shared dataclass is available for standardized configuration:

```python
@dataclass
class AttentionConfig:
    num_heads: int = 8
    head_dim: int = 32
    dropout: float = 0.0
    bias: bool = True
```

You can use `BaseAttention.from_config(cfg)` to construct instances from this config.

## Built-in implementations

The `ATTENTION_REGISTRY` ships with:

| Name | Description |
|------|-------------|
| `standard` | Scaled dot-product attention |
| `flash` | FlashAttention v2 (requires `flash_attn`) |
| `linear` | Linear attention via kernel feature maps |
| `gated` | Gated attention with sigmoid-gated output |

## Example: Sliding Window Attention

Let's implement a sliding window attention module that restricts each query to attend only to a local window of keys.

### Step 1: Create the module file

Create `molfun/modules/attention/sliding_window.py`:

```python
"""Sliding window (local) attention mechanism."""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from molfun.modules.attention.base import BaseAttention, ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("sliding_window")
class SlidingWindowAttention(BaseAttention):
    """
    Local attention where each query attends only to keys within
    a fixed window centered on its position.

    This reduces memory from O(L^2) to O(L * W) and is useful for
    very long protein sequences where global attention is prohibitive.

    Args:
        num_heads: Number of attention heads.
        head_dim: Dimension of each head.
        window_size: Number of positions to attend to on each side.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 32,
        window_size: int = 64,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, Lq, D = q.shape
        Lk = k.shape[2]

        # Standard scaled dot-product attention scores
        scale = D ** -0.5
        scores = torch.matmul(q * scale, k.transpose(-2, -1))  # [B, H, Lq, Lk]

        # Build a sliding window mask: attend only within window_size
        q_idx = torch.arange(Lq, device=q.device).unsqueeze(1)  # [Lq, 1]
        k_idx = torch.arange(Lk, device=q.device).unsqueeze(0)  # [1, Lk]
        window_mask = (q_idx - k_idx).abs() <= self.window_size  # [Lq, Lk]
        scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply optional additive bias (e.g., pair representation bias)
        if bias is not None:
            scores = scores + bias

        # Apply optional boolean mask
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        return torch.matmul(weights, v)  # [B, H, Lq, D]

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim
```

### Step 2: Ensure auto-import

Make sure your module is imported when the attention package loads. Add it to `molfun/modules/attention/__init__.py`:

```python
from molfun.modules.attention.sliding_window import SlidingWindowAttention  # noqa: F401
```

## Testing

Write tests in `tests/modules/attention/test_sliding_window.py`:

```python
import pytest
import torch

from molfun.modules.attention.base import ATTENTION_REGISTRY


class TestSlidingWindowAttention:
    """Tests for the sliding window attention module."""

    @pytest.fixture
    def attn(self):
        return ATTENTION_REGISTRY.build(
            "sliding_window", num_heads=4, head_dim=16, window_size=3
        )

    def test_registry_lookup(self):
        """Module is discoverable via the registry."""
        assert "sliding_window" in ATTENTION_REGISTRY
        cls = ATTENTION_REGISTRY.get("sliding_window")
        assert cls is not None

    def test_output_shape(self, attn):
        """Output shape matches [B, H, Lq, D]."""
        B, H, L, D = 2, 4, 10, 16
        q = k = v = torch.randn(B, H, L, D)
        out = attn(q, k, v)
        assert out.shape == (B, H, L, D)

    def test_window_locality(self, attn):
        """Queries only attend to keys within the window."""
        B, H, L, D = 1, 4, 20, 16
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)

        # Set all values to zero except at position 10
        v = torch.zeros(B, H, L, D)
        v[:, :, 10, :] = 1.0

        out = attn(q, k, v)

        # Positions far from 10 (outside window_size=3) should be ~zero
        assert out[:, :, 0, :].abs().max() < 1e-5
        assert out[:, :, -1, :].abs().max() < 1e-5
        # Position 10 itself should be non-zero
        assert out[:, :, 10, :].abs().max() > 0.01

    def test_with_mask_and_bias(self, attn):
        """forward works with mask and bias arguments."""
        B, H, L, D = 2, 4, 10, 16
        q = k = v = torch.randn(B, H, L, D)
        mask = torch.ones(B, 1, L, L, dtype=torch.bool)
        bias = torch.zeros(B, 1, L, L)
        out = attn(q, k, v, mask=mask, bias=bias)
        assert out.shape == (B, H, L, D)

    def test_properties(self, attn):
        assert attn.num_heads == 4
        assert attn.head_dim == 16
        assert attn.embed_dim == 64
```

Run the tests:

```bash
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/modules/attention/test_sliding_window.py -v
```

## Integration: Using your attention module

### With ModelBuilder

```python
from molfun.modules.builder import ModelBuilder

model = (
    ModelBuilder(d_single=256, d_pair=128)
    .embedder("input")
    .blocks("pairformer", num_blocks=8, attention="sliding_window", window_size=32)
    .structure_module("ipa")
    .build()
)
```

### With ModuleSwapper (replace attention in a pre-trained model)

```python
from molfun import MolfunStructureModel

model = MolfunStructureModel("openfold")
model.swap_attention("sliding_window", window_size=64)
```

### Direct instantiation via registry

```python
from molfun.modules.attention.base import ATTENTION_REGISTRY

attn = ATTENTION_REGISTRY.build("sliding_window", num_heads=8, head_dim=32, window_size=128)
```
