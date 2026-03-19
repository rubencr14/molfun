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

import molfun.backends.openfold.loss  # noqa: F401  registers openfold

# Trigger registration of built-in losses by importing their modules.
import molfun.losses.affinity  # noqa: F401  registers mse, mae, huber, pearson
from molfun.backends.openfold.loss import OpenFoldLoss
from molfun.losses.affinity import HuberLoss, MAELoss, MSELoss, PearsonLoss
from molfun.losses.base import LOSS_REGISTRY, LossFunction

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
]
