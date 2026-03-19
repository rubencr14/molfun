"""
Pluggable input embedders.

Embedders convert raw sequence/MSA inputs into initial representations
for the trunk blocks. Different models use different embedders:

- **InputEmbedder**: AF2-style (aatype one-hot + relpos + MSA embedding)
- **ESMEmbedder**: ESMFold-style (pre-trained language model features)
- **OneHotEmbedder**: Minimal baseline for testing

Registry
--------
    from molfun.modules.embedders import EMBEDDER_REGISTRY
    emb = EMBEDDER_REGISTRY.build("input", d_model=256, max_relpos=32)
"""

from molfun.modules.embedders.base import BaseEmbedder, EmbedderOutput, EMBEDDER_REGISTRY
from molfun.modules.embedders.input_embedder import InputEmbedder
from molfun.modules.embedders.esm_embedder import ESMEmbedder

__all__ = [
    "EMBEDDER_REGISTRY",
    "BaseEmbedder",
    "EmbedderOutput",
    "InputEmbedder",
    "ESMEmbedder",
]
