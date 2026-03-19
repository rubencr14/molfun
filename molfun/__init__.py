"""
Molfun — Fine-tuning, modular architecture and GPU acceleration
for molecular ML models.
"""

from molfun.models.structure import MolfunStructureModel
from molfun.backends.openfold import OpenFold
from molfun.predict import predict_structure, predict_properties, predict_affinity

__all__ = [
    "MolfunStructureModel",
    "OpenFold",
    "predict_structure",
    "predict_properties",
    "predict_affinity",
]
