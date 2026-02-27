"""
molfun.ml — machine learning for proteins.

Three levels of protein property prediction:

1. **Classical ML** (sequence features + sklearn)::

    from molfun.ml import ProteinRegressor
    reg = ProteinRegressor(estimator="random_forest")
    reg.fit(sequences, affinities)

2. **Property heads** (backbone embeddings + head)::

    from molfun.ml import PropertyHead
    head = PropertyHead(backbone="openfold", head_type="mlp")
    head.fit(pdb_paths, affinities)

3. **Structure fine-tuning** (FAPE loss, handled by backends)::

    # Uses molfun.pipelines with type: finetune
    # Targets = 3D coordinates from PDB (no external labels needed)
"""

from molfun.ml.features import (
    ProteinFeaturizer,
    AVAILABLE_FEATURES,
    DEFAULT_FEATURES,
)
from molfun.ml.estimators import (
    ProteinClassifier,
    ProteinRegressor,
    CLASSIFIER_NAMES,
    REGRESSOR_NAMES,
)
from molfun.ml.heads import PropertyHead
from molfun.ml.io import save_model, load_model

__all__ = [
    "ProteinFeaturizer",
    "ProteinClassifier",
    "ProteinRegressor",
    "PropertyHead",
    "AVAILABLE_FEATURES",
    "DEFAULT_FEATURES",
    "CLASSIFIER_NAMES",
    "REGRESSOR_NAMES",
    "save_model",
    "load_model",
]
