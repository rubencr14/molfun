"""
molfun.backends — model backend registry.

Each backend (OpenFold, Protenix, ESMFold, ...) lives in its own subpackage
and groups adapter, featurizer, loss, and helpers together.

Usage::

    from molfun.backends.openfold import OpenFoldAdapter, OpenFoldLoss
    from molfun.backends import BACKEND_REGISTRY
"""

from molfun.backends.base import BACKEND_REGISTRY, BackendSpec

__all__ = ["BACKEND_REGISTRY", "BackendSpec"]
