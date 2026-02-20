"""
Structure prediction loss head.

Wraps OpenFold's AlphaFoldLoss so that fine-tuning can optimize FAPE,
supervised chi, masked-MSA, distogram and pLDDT directly — without any
downstream affinity label.

Usage:
    head = StructureLossHead(config)          # config.loss from model_config()
    head = StructureLossHead.fape_only(config) # disable aux losses
    head = StructureLossHead.with_weights(    # custom term weights
        config, fape=1.0, supervised_chi=1.0, masked_msa=0.0
    )
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import torch
import torch.nn as nn

if TYPE_CHECKING:
    import ml_collections


class StructureLossHead(nn.Module):
    """
    Computes OpenFold's composite structure loss.

    Unlike AffinityHead, this head does NOT predict a scalar from the trunk
    output. Instead it evaluates FAPE + auxiliary losses directly against
    ground truth coordinates embedded in the input batch.

    The training loop must supply batches that include ground truth fields:
      backbone_rigid_tensor, backbone_rigid_mask, atom14_gt_positions,
      atom14_gt_exists, atom14_alt_gt_positions, atom14_alt_gt_exists,
      atom14_atom_is_ambiguous, atom14_atom_exists, chi_angles_sin_cos,
      chi_mask, seq_length, aatype, pseudo_beta, pseudo_beta_mask,
      residx_atom14_to_atom37, etc.

    Use OpenFoldFeaturizer to produce batches with all required fields.
    """

    def __init__(
        self,
        loss_config: "ml_collections.ConfigDict",
        disable_masked_msa: bool = True,
        disable_experimentally_resolved: bool = True,
    ):
        """
        Args:
            loss_config: ``config.loss`` sub-config from ``model_config()``.
            disable_masked_msa: Disable masked-MSA loss term (default True).
                Masked-MSA requires BERT-style masking in the data pipeline
                (``true_msa``, ``bert_mask`` fields). Set to False only if
                your data pipeline generates these fields.
            disable_experimentally_resolved: Disable the experimentally-resolved
                loss term (default True). This term requires a ``resolution``
                field from PDB headers. Not applicable for all structures.
        """
        super().__init__()
        try:
            from openfold.utils.loss import AlphaFoldLoss
        except ImportError:
            raise ImportError(
                "OpenFold is required: pip install git+https://github.com/aqlaboratory/openfold"
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
    def fape_only(cls, config) -> "StructureLossHead":
        """
        Only FAPE + supervised chi. No masked-MSA, distogram or pLDDT terms.
        Useful when you don't have full MSA features.
        """
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
    def with_weights(cls, config, **weights) -> "StructureLossHead":
        """
        Override individual loss term weights.

        Example:
            StructureLossHead.with_weights(config, fape=1.0, masked_msa=0.0)
        """
        import copy
        cfg = copy.deepcopy(config.loss)
        for term, w in weights.items():
            if hasattr(cfg, term):
                cfg[term].weight = w
        return cls(cfg)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        trunk_output,
        mask: Optional[torch.Tensor] = None,
        batch: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Compute structure loss.

        Args:
            trunk_output: TrunkOutput — extra["_raw_outputs"] must contain the
                          full OpenFold output dict (sm, distogram_logits, …).
            mask: Unused (kept for API compatibility with AffinityHead).
            batch: Feature dict with ground truth fields. Required.

        Returns:
            Scalar loss tensor (differentiable).
        """
        if batch is None:
            raise ValueError(
                "StructureLossHead requires `batch` (ground-truth feature dict). "
                "Make sure to call model.forward(batch, batch=batch)."
            )

        raw_outputs = trunk_output.extra.get("_raw_outputs")
        if raw_outputs is None:
            raise RuntimeError(
                "TrunkOutput.extra['_raw_outputs'] is missing. "
                "Ensure OpenFoldAdapter stores full outputs (use updated adapter)."
            )

        # OpenFold's loss functions expect tensors without the trailing recycling
        # dimension R.  The input batch has shape [..., R]; strip it by taking
        # index 0 on the last dim for every tensor that has one.
        batch_no_recycle = _strip_recycling_dim(batch)
        batch_no_recycle = _fill_missing_batch_fields(batch_no_recycle)

        # AlphaFoldLoss unconditionally calls find_structural_violations which
        # reads a resource file that may not be present (pip-installed OpenFold).
        # Pre-populate "violation" with a zero dict so the check is skipped.
        # The violation loss term has weight=0.0 by default anyway.
        raw_out = raw_outputs
        if "violation" not in raw_out:
            raw_out = dict(raw_out)
            raw_out["violation"] = _make_zero_violation(raw_out["final_atom_positions"])

        return self._loss_fn(raw_out, batch_no_recycle)

    def loss(
        self,
        preds: torch.Tensor,
        targets=None,
        loss_fn: str = "mse",
    ) -> dict[str, torch.Tensor]:
        """
        Compatibility wrapper: preds is already the scalar structure loss.

        The training loop calls head.loss(result["preds"], targets).
        For structure training, preds = scalar loss, targets = None.
        """
        return {"structure_loss": preds}

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        terms = {}
        for term in ("fape", "supervised_chi", "masked_msa", "distogram",
                      "plddt_loss", "experimentally_resolved"):
            try:
                terms[term] = float(self._loss_config[term].weight)
            except Exception:
                pass
        return {"type": "StructureLossHead", "loss_weights": terms}


# ======================================================================
# Helpers
# ======================================================================

def _fill_missing_batch_fields(batch: dict) -> dict:
    """
    Add optional batch fields that some OpenFold loss terms require but
    may not always be present in user-provided batches.

    AlphaFoldLoss calls ALL loss functions regardless of their weight, so
    we must provide valid (zero) fallback tensors for optional terms like
    masked-MSA (needs true_msa/bert_mask) and experimentally-resolved
    (needs resolution).
    """
    import torch
    batch = dict(batch)

    ref = batch.get("aatype")
    device = ref.device if ref is not None else torch.device("cpu")
    B = ref.shape[0] if (ref is not None and ref.dim() > 0) else 1

    # experimentally_resolved_loss
    if "resolution" not in batch:
        batch["resolution"] = torch.zeros(B, device=device)

    # masked_msa_loss: needs true_msa and bert_mask
    if "true_msa" not in batch:
        msa = batch.get("msa")
        if msa is not None:
            batch["true_msa"] = msa.clone()
        elif "msa_feat" in batch:
            # msa_feat shape [B, N, L, 49] → derive msa shape [B, N, L]
            s = batch["msa_feat"].shape
            batch["true_msa"] = torch.zeros(s[0], s[1], s[2], dtype=torch.long, device=device)
        else:
            batch["true_msa"] = torch.zeros(B, 1, 1, dtype=torch.long, device=device)
    if "bert_mask" not in batch:
        batch["bert_mask"] = torch.zeros_like(batch["true_msa"].float())

    return batch


def _make_zero_violation(ref: "torch.Tensor") -> dict:
    """
    Build a zero-filled violation dict matching find_structural_violations output.
    Used to skip the call when the required resource file is unavailable.
    """
    import torch
    z1 = ref.new_zeros(())          # scalar
    B, L = ref.shape[:2]
    z_BL   = ref.new_zeros(B, L)          # (B, L)
    z_BL14 = ref.new_zeros(B, L, 14)      # (B, L, 14)
    return {
        "between_residues": {
            "bonds_c_n_loss_mean":                  z1,
            "angles_ca_c_n_loss_mean":              z1,
            "angles_c_n_ca_loss_mean":              z1,
            "connections_per_residue_loss_sum":     z_BL,
            "connections_per_residue_violation_mask": z_BL,
            "clashes_mean_loss":                    z1,
            "clashes_per_atom_loss_sum":            z_BL14,
            "clashes_per_atom_clash_mask":          z_BL14,
            "clashes_per_atom_num_clash":           z_BL14,
        },
        "within_residues": {
            "per_atom_loss_sum":   z_BL14,
            "per_atom_violations": z_BL14,
            "per_atom_num_clash":  z_BL14,
        },
        "total_per_residue_violations_mask": z_BL,
    }


def _strip_recycling_dim(batch: dict) -> dict:
    """
    Remove the trailing recycling dimension R from all tensor values in batch.

    OpenFold's input tensors have shape [..., R] where R is the number of
    recycling iterations.  The loss functions expect [...] (R already consumed).
    We take index 0 on the last dim; for the common case R=1 this is a no-op
    shape change (squeeze).
    """
    import torch
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            # Only strip if last dim is 1 (single recycling pass) or >1
            # (take index 0 = last recycling state).  Leave scalars alone.
            out[k] = v[..., 0]
        else:
            out[k] = v
    return out
