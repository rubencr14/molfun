"""
Generic module registry for pluggable components.

Provides a type-safe, decorator-based registration pattern used
across all Molfun module families (attention, blocks, structure
modules, embedders, losses, etc.).

Usage
-----
    from molfun.modules.registry import ModuleRegistry

    ATTENTION_REGISTRY = ModuleRegistry("attention", BaseAttention)

    @ATTENTION_REGISTRY.register("flash")
    class FlashAttention(BaseAttention): ...

    # Later:
    attn = ATTENTION_REGISTRY.build("flash", num_heads=8, head_dim=32)
"""

from __future__ import annotations
from typing import Any, Optional
import torch.nn as nn


class ModuleRegistry:
    """
    Name → class registry with optional build-time validation.

    Each registry instance is scoped to a single module family
    (attention, block, structure_module, ...) and can enforce that
    registered classes inherit from a given base.
    """

    def __init__(self, name: str, base_class: Optional[type] = None):
        """
        Args:
            name: Human-readable family name (for error messages).
            base_class: If set, ``register()`` will reject classes
                        that don't subclass this type.
        """
        self.name = name
        self.base_class = base_class
        self._registry: dict[str, type[nn.Module]] = {}

    def register(self, name: str):
        """
        Class decorator::

            @REGISTRY.register("my_module")
            class MyModule(BaseModule): ...
        """
        def decorator(cls: type):
            if self.base_class is not None and not issubclass(cls, self.base_class):
                raise TypeError(
                    f"Cannot register '{name}' in {self.name} registry: "
                    f"{cls.__name__} does not inherit from {self.base_class.__name__}"
                )
            if name in self._registry:
                existing = self._registry[name].__name__
                raise ValueError(
                    f"Duplicate registration in {self.name} registry: "
                    f"'{name}' is already bound to {existing}"
                )
            self._registry[name] = cls
            return cls
        return decorator

    def build(self, name: str, **kwargs: Any) -> nn.Module:
        """Instantiate a registered module by name with keyword arguments."""
        cls = self[name]
        return cls(**kwargs)

    def get(self, name: str) -> Optional[type[nn.Module]]:
        """Return the class or None (no KeyError)."""
        return self._registry.get(name)

    def list(self) -> list[str]:
        """Return sorted list of registered names."""
        return sorted(self._registry)

    # ── dict-like access ──────────────────────────────────────────────

    def __getitem__(self, name: str) -> type[nn.Module]:
        if name not in self._registry:
            available = sorted(self._registry)
            raise KeyError(
                f"Module '{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __iter__(self):
        return iter(self._registry)

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"ModuleRegistry({self.name}, {sorted(self._registry)})"
