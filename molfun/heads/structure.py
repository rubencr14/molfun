"""
Structure prediction loss head.

Uses OpenFoldLoss from molfun.losses to optimize FAPE, supervised chi,
masked-MSA, distogram and pLDDT — without any downstream affinity label.

Usage
-----
    head = StructureLossHead(config)
    head = StructureLossHead.fape_only(config)
    head = StructureLossHead.with_weights(config, fape=1.0, masked_msa=0.0)
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import torch.nn as nn

from molfun.backends.openfold.loss import OpenFoldLoss

if TYPE_CHECKING:
    import torch
    import ml_collections


class StructureLossHead(nn.Module):
    """
    Head for structure prediction fine-tuning.

    Unlike AffinityHead, this module does NOT learn any parameters.
    It delegates entirely to OpenFoldLoss, which evaluates FAPE and
    auxiliary losses against ground truth coordinates embedded in the
    input batch.

    The training loop must supply batches produced by OpenFoldFeaturizer
    (they include all required ground truth fields).
    """

    def __init__(
        self,
        loss_config: "ml_collections.ConfigDict",
        disable_masked_msa: bool = True,
        disable_experimentally_resolved: bool = True,
    ):
        super().__init__()
        self._loss = OpenFoldLoss(
            loss_config,
            disable_masked_msa=disable_masked_msa,
            disable_experimentally_resolved=disable_experimentally_resolved,
        )

    # ------------------------------------------------------------------
    # Convenience constructors (delegate to OpenFoldLoss)
    # ------------------------------------------------------------------

    @classmethod
    def fape_only(cls, config) -> "StructureLossHead":
        """FAPE + supervised chi only."""
        head = cls.__new__(cls)
        nn.Module.__init__(head)
        head._loss = OpenFoldLoss.fape_only(config)
        return head

    @classmethod
    def with_weights(cls, config, **weights) -> "StructureLossHead":
        """Override individual loss-term weights."""
        head = cls.__new__(cls)
        nn.Module.__init__(head)
        head._loss = OpenFoldLoss.with_weights(config, **weights)
        return head

    # ------------------------------------------------------------------
    # Head interface
    # ------------------------------------------------------------------

    def forward(
        self,
        trunk_output,
        mask: Optional["torch.Tensor"] = None,
        batch: Optional[dict] = None,
    ) -> "torch.Tensor":
        """
        Compute structure loss from trunk output + ground truth batch.

        Args:
            trunk_output: TrunkOutput whose extra["_raw_outputs"] holds the
                          full OpenFold output dict.
            mask: Unused (kept for API compatibility with AffinityHead).
            batch: Feature dict with ground truth fields. Required.

        Returns:
            Scalar loss tensor (differentiable).
        """
        if batch is None:
            raise ValueError(
                "StructureLossHead requires `batch` (ground-truth feature dict)."
            )
        raw_outputs = trunk_output.extra.get("_raw_outputs")
        if raw_outputs is None:
            raise RuntimeError(
                "TrunkOutput.extra['_raw_outputs'] is missing. "
                "Ensure OpenFoldAdapter stores full outputs."
            )
        result = self._loss(raw_outputs, batch=batch)
        return result["structure_loss"]

    def loss(
        self,
        preds: "torch.Tensor",
        targets=None,
        loss_fn: str = "openfold",
    ) -> dict[str, "torch.Tensor"]:
        """
        Compatibility wrapper for the training loop.

        preds is already the scalar structure loss returned by forward().
        """
        return {"structure_loss": preds}

    def describe(self) -> dict:
        return self._loss.describe()
