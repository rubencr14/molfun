"""
Pluggable pair operations (triangular attention, triangular multiplication, outer product mean).

These are the pair-track building blocks used inside EvoformerBlock and
PairformerBlock. Making them pluggable allows researchers to experiment
with alternative pair update mechanisms.

Implementations are currently inlined in the block modules. This package
provides the registry and base class for future standalone implementations.
"""

from molfun.modules.registry import ModuleRegistry

PAIR_OP_REGISTRY = ModuleRegistry("pair_op")
