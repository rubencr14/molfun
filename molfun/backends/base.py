"""
Backend specification: groups adapter, featurizer, loss and helpers for a model.

Each backend (OpenFold, Protenix, ESMFold, ...) registers a BackendSpec so
the rest of molfun can discover its components by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class BackendSpec:
    """All components that a model backend provides."""

    name: str
    adapter_cls: type
    loss_cls: type | None = None
    featurizer_cls: type | None = None
    helpers_module: object | None = None
    convenience_cls: type | None = None


class BackendRegistry:
    """
    Name → BackendSpec registry.

    Usage::

        BACKEND_REGISTRY.register(BackendSpec(name="openfold", ...))
        spec = BACKEND_REGISTRY["openfold"]
        adapter = spec.adapter_cls(config=cfg)
    """

    def __init__(self):
        self._registry: dict[str, BackendSpec] = {}

    def register(self, spec: BackendSpec) -> None:
        self._registry[spec.name] = spec

    def __getitem__(self, name: str) -> BackendSpec:
        if name not in self._registry:
            raise KeyError(f"Backend '{name}' not found. Available: {sorted(self._registry)}")
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __iter__(self):
        return iter(self._registry)

    def __repr__(self) -> str:
        return f"BackendRegistry({sorted(self._registry)})"


BACKEND_REGISTRY = BackendRegistry()
