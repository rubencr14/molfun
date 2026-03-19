"""
Molfun — Fine-tuning, modular architecture and GPU acceleration
for molecular ML models.
"""

from molfun.backends.openfold import OpenFold
from molfun.models.structure import MolfunStructureModel
from molfun.predict import predict_affinity, predict_properties, predict_structure

__all__ = [
    "MolfunStructureModel",
    "OpenFold",
    "predict_structure",
    "predict_properties",
    "predict_affinity",
]
