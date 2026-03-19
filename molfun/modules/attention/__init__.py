"""
Pluggable attention mechanisms.

Registry
--------
    from molfun.modules.attention import ATTENTION_REGISTRY

    attn = ATTENTION_REGISTRY.build("standard", num_heads=8, head_dim=32)
    attn = ATTENTION_REGISTRY.build("flash", num_heads=8, head_dim=32)

Custom
------
    @ATTENTION_REGISTRY.register("my_attn")
    class MyAttention(BaseAttention): ...
"""

from molfun.modules.attention.base import BaseAttention, ATTENTION_REGISTRY
from molfun.modules.attention.standard import StandardAttention
from molfun.modules.attention.flash import FlashAttention
from molfun.modules.attention.linear import LinearAttention
from molfun.modules.attention.gated import GatedAttention

__all__ = [
    "ATTENTION_REGISTRY",
    "BaseAttention",
    "StandardAttention",
    "FlashAttention",
    "LinearAttention",
    "GatedAttention",
]
