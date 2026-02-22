"""
molfun.helpers — shared utility functions.

Training utilities
------------------
    EMA               Exponential Moving Average of model parameters
    build_scheduler   LR scheduler: linear warmup + cosine/linear/constant decay
    unpack_batch      Normalize DataLoader output → (features, targets, mask)
    to_device         Move batch tensors to a device

OpenFold batch helpers
----------------------
    strip_recycling_dim       Remove trailing recycling dim R from tensor dict
    fill_missing_batch_fields Add zero fallbacks for optional AlphaFoldLoss fields
    make_zero_violation       Build zero violation dict to skip stereo resource check
"""

from molfun.helpers.training import EMA, build_scheduler, unpack_batch, to_device
from molfun.helpers.openfold import (
    strip_recycling_dim,
    fill_missing_batch_fields,
    make_zero_violation,
)

__all__ = [
    # Training
    "EMA",
    "build_scheduler",
    "unpack_batch",
    "to_device",
    # OpenFold
    "strip_recycling_dim",
    "fill_missing_batch_fields",
    "make_zero_violation",
]
