# Adding Embedders

Embedders convert raw input features (amino acid types, residue indices, MSA alignments) into the initial single and pair representations that feed into the trunk blocks. They are the first learnable component in the model pipeline.

## The BaseEmbedder interface

All embedders inherit from `BaseEmbedder` in `molfun/modules/embedders/base.py`:

```python
class BaseEmbedder(ABC, nn.Module):
    """Converts raw features -> initial representations for trunk blocks."""

    @abstractmethod
    def forward(
        self,
        aatype: torch.Tensor,                          # [B, L] int64, 0-20
        residue_index: torch.Tensor,                    # [B, L]
        msa: Optional[torch.Tensor] = None,             # [B, N, L, D_msa_feat]
        msa_mask: Optional[torch.Tensor] = None,        # [B, N, L]
        **kwargs,
    ) -> EmbedderOutput:
        ...

    @property
    @abstractmethod
    def d_single(self) -> int:
        """Output single representation dimension."""

    @property
    @abstractmethod
    def d_pair(self) -> int:
        """Output pair representation dimension (0 if no pair track)."""
```

### EmbedderOutput

```python
@dataclass
class EmbedderOutput:
    single: torch.Tensor                          # [B, L, D_s]
    pair: Optional[torch.Tensor] = None           # [B, L, L, D_p]
```

## Built-in implementations

| Name | Description |
|------|-------------|
| `input` | AF2-style: learned amino acid embeddings + relative position encoding + outer product for pair |
| `esm_embedder` | Uses a frozen ESM protein language model to produce single representations |

## Example: ProtTrans Embedder

Let's implement an embedder that uses a ProtTrans (ProtBERT/ProtT5) model to generate initial representations. This demonstrates how to wrap a pre-trained language model as a Molfun embedder.

### Step 1: Create the module file

Create `molfun/modules/embedders/prottrans.py`:

```python
"""
ProtTrans embedder: uses a frozen ProtTrans model to generate
initial single representations, with a learned pair projection.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from molfun.modules.embedders.base import BaseEmbedder, EmbedderOutput, EMBEDDER_REGISTRY


@EMBEDDER_REGISTRY.register("prottrans")
class ProtTransEmbedder(BaseEmbedder):
    """
    Generates single representations from a pre-trained ProtTrans model
    and derives pair representations via outer product + linear projection.

    The ProtTrans model is kept frozen by default. Only the projection
    layers and pair construction are trainable.

    Args:
        d_single: Target single representation dimension.
        d_pair: Target pair representation dimension.
        prottrans_dim: Hidden dimension of the ProtTrans model (1024 for ProtBERT).
        num_aa_types: Number of amino acid types (21 = 20 standard + unknown).
        max_relpos: Maximum relative position for positional encoding.
        freeze_lm: Whether to freeze the language model weights.
    """

    def __init__(
        self,
        d_single: int = 256,
        d_pair: int = 128,
        prottrans_dim: int = 1024,
        num_aa_types: int = 21,
        max_relpos: int = 32,
        freeze_lm: bool = True,
    ):
        super().__init__()
        self._d_single = d_single
        self._d_pair = d_pair
        self.freeze_lm = freeze_lm

        # Fallback embedding for when ProtTrans is not available
        self.aa_embed = nn.Embedding(num_aa_types, prottrans_dim)

        # Project LM output to target single dimension
        self.single_proj = nn.Sequential(
            nn.LayerNorm(prottrans_dim),
            nn.Linear(prottrans_dim, d_single),
            nn.GELU(),
            nn.Linear(d_single, d_single),
        )

        # Relative position encoding for pair representation
        self.relpos_embed = nn.Embedding(2 * max_relpos + 1, d_pair)
        self.max_relpos = max_relpos

        # Outer product projection: single x single -> pair
        self.outer_proj = nn.Linear(d_single * 2, d_pair)
        self.pair_norm = nn.LayerNorm(d_pair)

        self._lm = None  # Lazy-loaded ProtTrans model

    def _get_lm_embeddings(self, aatype: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings from ProtTrans model, falling back to learned
        embeddings if the model is not available.
        """
        if self._lm is not None:
            with torch.no_grad() if self.freeze_lm else torch.enable_grad():
                return self._lm(aatype)

        # Fallback: use learned amino acid embeddings
        return self.aa_embed(aatype)

    def forward(
        self,
        aatype: torch.Tensor,
        residue_index: torch.Tensor,
        msa: Optional[torch.Tensor] = None,
        msa_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> EmbedderOutput:
        B, L = aatype.shape

        # Single representation from LM
        lm_out = self._get_lm_embeddings(aatype)    # [B, L, prottrans_dim]
        single = self.single_proj(lm_out)            # [B, L, d_single]

        # Pair representation from outer product + relative position
        si = single.unsqueeze(2).expand(-1, -1, L, -1)    # [B, L, L, D_s]
        sj = single.unsqueeze(1).expand(-1, L, -1, -1)    # [B, L, L, D_s]
        outer = self.outer_proj(torch.cat([si, sj], dim=-1))  # [B, L, L, D_p]

        # Relative position encoding
        ri = residue_index.unsqueeze(2)  # [B, L, 1]
        rj = residue_index.unsqueeze(1)  # [B, 1, L]
        relpos = (ri - rj).clamp(-self.max_relpos, self.max_relpos) + self.max_relpos
        relpos_feat = self.relpos_embed(relpos)  # [B, L, L, D_p]

        pair = self.pair_norm(outer + relpos_feat)  # [B, L, L, D_p]

        return EmbedderOutput(single=single, pair=pair)

    @property
    def d_single(self) -> int:
        return self._d_single

    @property
    def d_pair(self) -> int:
        return self._d_pair

    def load_prottrans(self, model_name: str = "Rostlab/prot_bert") -> None:
        """
        Load a ProtTrans model from HuggingFace.

        Requires ``transformers`` to be installed. This is lazy-loaded
        to avoid importing heavy dependencies at module registration time.
        """
        from transformers import AutoModel
        self._lm = AutoModel.from_pretrained(model_name)
        if self.freeze_lm:
            for p in self._lm.parameters():
                p.requires_grad = False
```

### Step 2: Register via __init__.py

Add the import to `molfun/modules/embedders/__init__.py`:

```python
from molfun.modules.embedders.prottrans import ProtTransEmbedder  # noqa: F401
```

## Testing

Create `tests/modules/embedders/test_prottrans.py`:

```python
import pytest
import torch

from molfun.modules.embedders.base import EMBEDDER_REGISTRY, EmbedderOutput


class TestProtTransEmbedder:

    @pytest.fixture
    def embedder(self):
        return EMBEDDER_REGISTRY.build(
            "prottrans", d_single=64, d_pair=32, prottrans_dim=128
        )

    def test_registry_lookup(self):
        assert "prottrans" in EMBEDDER_REGISTRY

    def test_output_type(self, embedder):
        aatype = torch.randint(0, 21, (2, 10))
        residue_index = torch.arange(10).unsqueeze(0).expand(2, -1)
        out = embedder(aatype, residue_index)
        assert isinstance(out, EmbedderOutput)

    def test_single_shape(self, embedder):
        B, L = 2, 15
        aatype = torch.randint(0, 21, (B, L))
        residue_index = torch.arange(L).unsqueeze(0).expand(B, -1)
        out = embedder(aatype, residue_index)
        assert out.single.shape == (B, L, 64)

    def test_pair_shape(self, embedder):
        B, L = 2, 15
        aatype = torch.randint(0, 21, (B, L))
        residue_index = torch.arange(L).unsqueeze(0).expand(B, -1)
        out = embedder(aatype, residue_index)
        assert out.pair.shape == (B, L, L, 32)

    def test_properties(self, embedder):
        assert embedder.d_single == 64
        assert embedder.d_pair == 32

    def test_gradient_flow(self, embedder):
        """Projection layers should receive gradients."""
        aatype = torch.randint(0, 21, (2, 10))
        residue_index = torch.arange(10).unsqueeze(0).expand(2, -1)
        out = embedder(aatype, residue_index)
        loss = out.single.sum() + out.pair.sum()
        loss.backward()
        assert embedder.single_proj[1].weight.grad is not None

    def test_relpos_clipping(self, embedder):
        """Relative positions beyond max_relpos are clipped, not erroring."""
        B, L = 1, 100  # long sequence to exceed max_relpos=32
        aatype = torch.randint(0, 21, (B, L))
        residue_index = torch.arange(L).unsqueeze(0)
        out = embedder(aatype, residue_index)
        assert out.pair.shape == (B, L, L, 32)
```

Run the tests:

```bash
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/modules/embedders/test_prottrans.py -v
```

## Integration

### With ModelBuilder

```python
from molfun.modules.builder import ModelBuilder

model = (
    ModelBuilder(d_single=256, d_pair=128)
    .embedder("prottrans", prottrans_dim=1024)
    .blocks("pairformer", num_blocks=8)
    .structure_module("ipa")
    .build()
)
```

### Loading the actual ProtTrans weights

```python
from molfun.modules.embedders.base import EMBEDDER_REGISTRY

embedder = EMBEDDER_REGISTRY.build("prottrans", d_single=256, d_pair=128)
embedder.load_prottrans("Rostlab/prot_bert")
```

> **Note:** The `load_prottrans` method requires `transformers` to be installed. The embedder works without it using learned amino acid embeddings as a fallback, which is sufficient for testing and prototyping.
