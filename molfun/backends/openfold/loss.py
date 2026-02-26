"""
OpenFold structure loss.

Wraps OpenFold's AlphaFoldLoss so it fits the LossFunction interface.

Usage
-----
from molfun.backends.openfold import OpenFoldLoss

loss_fn = OpenFoldLoss(config.loss)
loss_fn = OpenFoldLoss.fape_only(config)
loss_fn = OpenFoldLoss.with_weights(config, fape=1.0, masked_msa=0.0)

scalar = loss_fn(raw_openfold_outputs, batch=feature_dict)
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import torch

from molfun.losses.base import LOSS_REGISTRY, LossFunction
from molfun.backends.openfold.helpers import (
    strip_recycling_dim,
    fill_missing_batch_fields,
    make_zero_violation,
)

if TYPE_CHECKING:
    import ml_collections


@LOSS_REGISTRY.register("openfold")
class OpenFoldLoss(LossFunction):
    """
    Composite structure loss that wraps OpenFold's AlphaFoldLoss.

    Loss terms (weights configurable via config.loss or with_weights()):
      - fape                    Frame Aligned Point Error (backbone + side-chain)
      - supervised_chi          Side-chain torsion angle loss
      - distogram               Pairwise Cβ-distance distribution loss
      - plddt_loss              Predicted lDDT confidence loss
      - masked_msa              Masked MSA reconstruction (disabled by default)
      - experimentally_resolved Experimentally resolved atom loss (disabled by default)
      - violation               Steric clash / bond geometry (disabled by default)

    Args:
        loss_config: ``config.loss`` sub-config from OpenFold's ``model_config()``.
        disable_masked_msa: Zero-out masked-MSA weight (default True).
            Enable only if the data pipeline produces ``true_msa`` / ``bert_mask``.
        disable_experimentally_resolved: Zero-out this term (default True).
            Enable only if PDB resolution metadata is present in the batch.
    """

    def __init__(
        self,
        loss_config: "ml_collections.ConfigDict",
        disable_masked_msa: bool = True,
        disable_experimentally_resolved: bool = True,
    ):
        super().__init__()
        try:
            from openfold.utils.loss import AlphaFoldLoss
        except ImportError:
            raise ImportError(
                "OpenFold is required: "
                "pip install git+https://github.com/aqlaboratory/openfold"
            )
        import copy
        loss_config = copy.deepcopy(loss_config)
        if disable_masked_msa:
            loss_config.masked_msa.weight = 0.0
        if disable_experimentally_resolved:
            loss_config.experimentally_resolved.weight = 0.0

        self._loss_fn = AlphaFoldLoss(loss_config)
        self._loss_config = loss_config

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def fape_only(cls, config) -> "OpenFoldLoss":
        """FAPE + supervised chi only — no MSA, distogram or pLDDT terms."""
        import copy
        cfg = copy.deepcopy(config.loss)
        cfg.masked_msa.weight = 0.0
        cfg.distogram.weight = 0.0
        cfg.experimentally_resolved.weight = 0.0
        cfg.plddt_loss.weight = 0.0
        if hasattr(cfg, "tm"):
            cfg.tm.weight = 0.0
        return cls(cfg)

    @classmethod
    def with_weights(cls, config, **weights) -> "OpenFoldLoss":
        """
        Override individual loss-term weights.

        Example::

            OpenFoldLoss.with_weights(config, fape=1.0, masked_msa=0.0)
        """
        import copy
        cfg = copy.deepcopy(config.loss)
        for term, w in weights.items():
            if hasattr(cfg, term):
                cfg[term].weight = w
        return cls(cfg)

    # ------------------------------------------------------------------
    # LossFunction interface
    # ------------------------------------------------------------------

    def forward(
        self,
        preds,
        targets: Optional[torch.Tensor] = None,
        batch: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute OpenFold structure loss.

        Args:
            preds: Raw OpenFold output dict (from TrunkOutput.extra["_raw_outputs"]).
                   Must contain keys: ``sm``, ``distogram_logits``,
                   ``final_atom_positions``, ``final_atom_mask``, etc.
            targets: Unused (ground truth is embedded in ``batch``).
            batch: Feature dict with ground truth fields produced by
                   ``OpenFoldFeaturizer``. Required.

        Returns:
            ``{"structure_loss": scalar_tensor}``
        """
        if batch is None:
            raise ValueError(
                "OpenFoldLoss requires `batch` (ground-truth feature dict). "
                "Pass batch=feature_dict when calling the loss."
            )
        if not isinstance(preds, dict):
            raise TypeError(
                "OpenFoldLoss expects `preds` to be the raw OpenFold output dict "
                "(TrunkOutput.extra['_raw_outputs'])."
            )

        batch_no_recycle = strip_recycling_dim(batch)
        batch_no_recycle = fill_missing_batch_fields(batch_no_recycle)

        raw_out = preds
        if "violation" not in raw_out:
            raw_out = dict(raw_out)
            raw_out["violation"] = make_zero_violation(raw_out["final_atom_positions"])

        scalar = self._loss_fn(raw_out, batch_no_recycle)
        return {"structure_loss": scalar}

    def describe(self) -> dict:
        """Return the active loss weights for logging / inspection."""
        terms = {}
        for term in (
            "fape", "supervised_chi", "masked_msa", "distogram",
            "plddt_loss", "experimentally_resolved",
        ):
            try:
                terms[term] = float(self._loss_config[term].weight)
            except Exception:
                pass
        return {"type": "OpenFoldLoss", "loss_weights": terms}
