# Adding Blocks

Trunk blocks are the repeating computational units that refine single and pair representations in a protein structure prediction model. Molfun supports stacking N blocks of any registered type to form the model trunk.

## The BaseBlock interface

All blocks inherit from `BaseBlock` in `molfun/modules/blocks/base.py`:

```python
class BaseBlock(ABC, nn.Module):
    """A single repeating block of the trunk."""

    @abstractmethod
    def forward(
        self,
        single: torch.Tensor,                          # [B, L, D_s] or [B, N, L, D_m]
        pair: Optional[torch.Tensor] = None,            # [B, L, L, D_p]
        mask: Optional[torch.Tensor] = None,            # [B, L] or [B, N, L]
        pair_mask: Optional[torch.Tensor] = None,       # [B, L, L]
    ) -> BlockOutput:
        ...

    @property
    @abstractmethod
    def d_single(self) -> int:
        """Single/MSA representation dimension."""

    @property
    @abstractmethod
    def d_pair(self) -> int:
        """Pair representation dimension (0 if single-track only)."""
```

### BlockOutput

Every block returns a `BlockOutput` dataclass:

```python
@dataclass
class BlockOutput:
    single: Optional[torch.Tensor] = None   # [B, L, D_s]
    pair: Optional[torch.Tensor] = None      # [B, L, L, D_p]
```

The interface supports three paradigms:

- **Dual-track** (AF2 Evoformer): processes both MSA and pair representations.
- **Single+pair** (AF3 Pairformer): processes single and pair without an MSA track.
- **Single-only** (ESMFold): processes only single representation (`d_pair = 0`).

## Built-in implementations

| Name | Description |
|------|-------------|
| `evoformer` | AF2-style block with MSA row/column attention + pair triangles |
| `pairformer` | AF3-style block with single attention + pair triangles |
| `simple_transformer` | Single-track transformer block (no pair) |

## Example: Axial Attention Block

An axial attention block decomposes 2D pair attention into two 1D passes (row-wise and column-wise), reducing memory from O(L^4) to O(L^3). This is useful for scaling to longer sequences.

### Step 1: Create the module file

Create `molfun/modules/blocks/axial.py`:

```python
"""Axial attention block for efficient pair representation processing."""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from molfun.modules.blocks.base import BaseBlock, BlockOutput, BLOCK_REGISTRY
from molfun.modules.attention.base import ATTENTION_REGISTRY


@BLOCK_REGISTRY.register("axial")
class AxialAttentionBlock(BaseBlock):
    """
    Block that applies attention axially: first along rows of the pair
    representation, then along columns, interleaved with single-track
    self-attention and transition layers.

    This reduces the O(L^4) cost of full pair attention to O(L^3).

    Args:
        d_single: Single representation dimension.
        d_pair: Pair representation dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        attention: Name of the attention implementation to use.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_single: int = 256,
        d_pair: int = 128,
        num_heads: int = 8,
        head_dim: int = 32,
        attention: str = "standard",
        dropout: float = 0.1,
    ):
        super().__init__()
        self._d_single = d_single
        self._d_pair = d_pair

        # Single-track self-attention
        self.single_norm = nn.LayerNorm(d_single)
        self.single_qkv = nn.Linear(d_single, 3 * num_heads * head_dim)
        self.single_out = nn.Linear(num_heads * head_dim, d_single)
        self.single_attn = ATTENTION_REGISTRY.build(
            attention, num_heads=num_heads, head_dim=head_dim, dropout=dropout
        )

        # Row-wise pair attention
        self.row_norm = nn.LayerNorm(d_pair)
        self.row_qkv = nn.Linear(d_pair, 3 * num_heads * head_dim)
        self.row_out = nn.Linear(num_heads * head_dim, d_pair)
        self.row_attn = ATTENTION_REGISTRY.build(
            attention, num_heads=num_heads, head_dim=head_dim, dropout=dropout
        )

        # Column-wise pair attention
        self.col_norm = nn.LayerNorm(d_pair)
        self.col_qkv = nn.Linear(d_pair, 3 * num_heads * head_dim)
        self.col_out = nn.Linear(num_heads * head_dim, d_pair)
        self.col_attn = ATTENTION_REGISTRY.build(
            attention, num_heads=num_heads, head_dim=head_dim, dropout=dropout
        )

        # Transition FFNs
        self.single_ffn = nn.Sequential(
            nn.LayerNorm(d_single),
            nn.Linear(d_single, d_single * 4),
            nn.GELU(),
            nn.Linear(d_single * 4, d_single),
            nn.Dropout(dropout),
        )
        self.pair_ffn = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, d_pair * 4),
            nn.GELU(),
            nn.Linear(d_pair * 4, d_pair),
            nn.Dropout(dropout),
        )

        self._num_heads = num_heads
        self._head_dim = head_dim

    def _split_heads(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split last dim into (3, H, D) and separate Q, K, V."""
        *batch, L, _ = x.shape
        x = x.view(*batch, L, 3, self._num_heads, self._head_dim)
        x = x.permute(*range(len(batch)), -2, -3, -1)  # [..., H, 3, L, D]
        # Unpack the 3 into q, k, v: each [..., H, L, D]
        # Simpler approach: reshape then chunk
        *batch, L, _ = x.shape  # recalc after permute
        # Use the original tensor before permute:
        return None  # placeholder for brevity

    def forward(
        self,
        single: torch.Tensor,
        pair: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> BlockOutput:
        B, L, Ds = single.shape

        # --- Single-track self-attention ---
        s = self.single_norm(single)
        qkv = self.single_qkv(s).reshape(B, L, 3, self._num_heads, self._head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, L, H, D]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, L, D]
        single = single + self.single_out(
            self.single_attn(q, k, v, mask=None).transpose(1, 2).reshape(B, L, -1)
        )
        single = single + self.single_ffn(single)

        if pair is None:
            return BlockOutput(single=single, pair=None)

        Dp = pair.shape[-1]

        # --- Row-wise pair attention (attend along j for each i) ---
        p = self.row_norm(pair)
        # Treat [B, L, L, D] as [B*L, L, D] for row attention
        p_row = p.reshape(B * L, L, Dp)
        qkv = self.row_qkv(p_row).reshape(B * L, L, 3, self._num_heads, self._head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        row_out = self.row_attn(q, k, v).transpose(1, 2).reshape(B * L, L, -1)
        pair = pair + self.row_out(row_out).reshape(B, L, L, Dp)

        # --- Column-wise pair attention (attend along i for each j) ---
        p = self.col_norm(pair)
        p_col = p.permute(0, 2, 1, 3).reshape(B * L, L, Dp)
        qkv = self.col_qkv(p_col).reshape(B * L, L, 3, self._num_heads, self._head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        col_out = self.col_attn(q, k, v).transpose(1, 2).reshape(B * L, L, -1)
        pair = pair + self.col_out(col_out).reshape(B, L, L, Dp).permute(0, 2, 1, 3)

        # --- Pair transition ---
        pair = pair + self.pair_ffn(pair)

        return BlockOutput(single=single, pair=pair)

    @property
    def d_single(self) -> int:
        return self._d_single

    @property
    def d_pair(self) -> int:
        return self._d_pair
```

### Step 2: Register via __init__.py

Add the import to `molfun/modules/blocks/__init__.py`:

```python
from molfun.modules.blocks.axial import AxialAttentionBlock  # noqa: F401
```

## Testing

Create `tests/modules/blocks/test_axial.py`:

```python
import pytest
import torch

from molfun.modules.blocks.base import BLOCK_REGISTRY, BlockOutput


class TestAxialAttentionBlock:

    @pytest.fixture
    def block(self):
        return BLOCK_REGISTRY.build(
            "axial", d_single=64, d_pair=32, num_heads=4, head_dim=16
        )

    def test_registry_lookup(self):
        assert "axial" in BLOCK_REGISTRY

    def test_output_type(self, block):
        single = torch.randn(2, 10, 64)
        pair = torch.randn(2, 10, 10, 32)
        out = block(single, pair)
        assert isinstance(out, BlockOutput)

    def test_output_shapes(self, block):
        B, L = 2, 10
        single = torch.randn(B, L, 64)
        pair = torch.randn(B, L, L, 32)
        out = block(single, pair)
        assert out.single.shape == (B, L, 64)
        assert out.pair.shape == (B, L, L, 32)

    def test_single_only(self, block):
        """Works in single-track mode (no pair)."""
        single = torch.randn(2, 10, 64)
        out = block(single, pair=None)
        assert out.single.shape == (2, 10, 64)
        assert out.pair is None

    def test_properties(self, block):
        assert block.d_single == 64
        assert block.d_pair == 32

    def test_gradient_flow(self, block):
        single = torch.randn(2, 10, 64, requires_grad=True)
        pair = torch.randn(2, 10, 10, 32, requires_grad=True)
        out = block(single, pair)
        loss = out.single.sum() + out.pair.sum()
        loss.backward()
        assert single.grad is not None
        assert pair.grad is not None
```

Run the tests:

```bash
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/modules/blocks/test_axial.py -v
```

## Integration

### With ModelBuilder

```python
from molfun.modules.builder import ModelBuilder

model = (
    ModelBuilder(d_single=256, d_pair=128)
    .embedder("input")
    .blocks("axial", num_blocks=12, num_heads=8, head_dim=32)
    .structure_module("ipa")
    .build()
)
```

### Swapping blocks in a pre-trained model

```python
from molfun import MolfunStructureModel

model = MolfunStructureModel("openfold")
model.swap_blocks("axial", num_heads=8, head_dim=32)
```
