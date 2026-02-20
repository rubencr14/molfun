"""
Model wrappers for fine-tuning and inference.
"""

from molfun.models.structure import MolfunStructureModel
from molfun.models.openfold import OpenFold

__all__ = ["MolfunStructureModel", "OpenFold"]
