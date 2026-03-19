"""
OpenFold backend — all OpenFold-specific components in one place.

Available exports:
    OpenFoldAdapter     Model adapter (wraps AlphaFold nn.Module)
    OpenFoldFeaturizer  PDB/mmCIF → full feature dict
    OpenFoldLoss        FAPE + auxiliary structure losses
    OpenFold            Convenience alias for MolfunStructureModel

    strip_recycling_dim, fill_missing_batch_fields, make_zero_violation
                        Batch pre-processing helpers

All imports are lazy to avoid circular dependencies at package init time.
"""

_LAZY_IMPORTS = {
    "OpenFoldAdapter":          ".adapter",
    "OpenFoldFeaturizer":       ".featurizer",
    "OpenFoldLoss":             ".loss",
    "OpenFold":                 ".convenience",
    "strip_recycling_dim":      ".helpers",
    "fill_missing_batch_fields": ".helpers",
    "make_zero_violation":      ".helpers",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
