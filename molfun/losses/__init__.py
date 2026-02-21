"""
molfun.losses — loss function registry and built-in implementations.

Built-in losses
---------------
Affinity (regression / ranking):
    "mse"      MSELoss      Mean squared error
    "mae"      MAELoss      Mean absolute error
    "huber"    HuberLoss    Huber / smooth-L1
    "pearson"  PearsonLoss  1 − Pearson correlation

Structure prediction:
    "openfold" OpenFoldLoss  FAPE + aux losses via AlphaFoldLoss

Quick start
-----------
>>> from molfun.losses import LOSS_REGISTRY
>>> loss_fn = LOSS_REGISTRY["huber"]()
>>> result  = loss_fn(preds, targets)   # {"affinity_loss": tensor}

>>> from molfun.losses import OpenFoldLoss
>>> loss_fn = OpenFoldLoss.fape_only(config)
>>> result  = loss_fn(raw_outputs, batch=feature_dict)  # {"structure_loss": tensor}

Registering a custom loss
--------------------------
>>> from molfun.losses import LOSS_REGISTRY, LossFunction
>>> @LOSS_REGISTRY.register("tmscore")
... class TMScoreLoss(LossFunction):
...     def forward(self, preds, targets=None, batch=None):
...         ...
"""

from molfun.losses.base import LOSS_REGISTRY, LossFunction

# Trigger registration of built-in losses by importing their modules.
import molfun.losses.affinity   # noqa: F401  registers mse, mae, huber, pearson
import molfun.losses.openfold   # noqa: F401  registers openfold

from molfun.losses.affinity import MSELoss, MAELoss, HuberLoss, PearsonLoss
from molfun.losses.openfold import (
    OpenFoldLoss,
    strip_recycling_dim,
    fill_missing_batch_fields,
    make_zero_violation,
)

__all__ = [
    # Registry + ABC
    "LOSS_REGISTRY",
    "LossFunction",
    # Affinity losses
    "MSELoss",
    "MAELoss",
    "HuberLoss",
    "PearsonLoss",
    # Structure losses
    "OpenFoldLoss",
    # Helpers (public for advanced use / testing)
    "strip_recycling_dim",
    "fill_missing_batch_fields",
    "make_zero_violation",
]
