"""
OpenFold batch pre-processing helpers.

Utilities for adapting feature dicts produced by OpenFoldFeaturizer so they
are compatible with AlphaFoldLoss and other OpenFold internals.

  strip_recycling_dim      Remove trailing recycling dimension R from tensors
  fill_missing_batch_fields  Add zero fallbacks for optional AlphaFoldLoss fields
  make_zero_violation      Build zero-filled violation dict to skip stereo check
"""

from __future__ import annotations
import torch


def strip_recycling_dim(batch: dict) -> dict:
    """
    Remove the trailing recycling dimension R from every tensor in the batch.

    OpenFold input tensors have shape ``[..., R]`` where R is the number of
    recycling iterations used during featurization (commonly R=1).
    ``AlphaFoldLoss`` expects tensors without this dimension (R already
    consumed by the model's recycling loop).

    Strategy: take index 0 on the last dim — for R=1 this is equivalent to
    a squeeze, for R>1 it picks the final recycling state.

    Args:
        batch: Feature dict as returned by ``OpenFoldFeaturizer``.

    Returns:
        New dict with the recycling dimension removed from all tensors.
    """
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            out[k] = v[..., 0]
        else:
            out[k] = v
    return out


def fill_missing_batch_fields(batch: dict) -> dict:
    """
    Add zero-filled fallback tensors for optional ``AlphaFoldLoss`` fields.

    ``AlphaFoldLoss`` calls every loss term regardless of its configured
    weight, so we must provide valid tensors for terms that may be disabled
    (masked-MSA needs ``true_msa`` / ``bert_mask``; experimentally-resolved
    needs ``resolution``).  Zero values are correct when the corresponding
    term weight is 0.0.

    Args:
        batch: Feature dict (recycling dim already stripped).

    Returns:
        New dict with ``resolution``, ``true_msa``, and ``bert_mask`` added
        if they were absent.
    """
    batch  = dict(batch)
    ref    = batch.get("aatype")
    device = ref.device if ref is not None else torch.device("cpu")
    B      = ref.shape[0] if (ref is not None and ref.dim() > 0) else 1

    if "resolution" not in batch:
        batch["resolution"] = torch.zeros(B, device=device)

    if "true_msa" not in batch:
        msa = batch.get("msa")
        if msa is not None:
            batch["true_msa"] = msa.clone()
        elif "msa_feat" in batch:
            s = batch["msa_feat"].shape          # [B, N_msa, L, 49]
            batch["true_msa"] = torch.zeros(
                s[0], s[1], s[2], dtype=torch.long, device=device
            )
        else:
            batch["true_msa"] = torch.zeros(B, 1, 1, dtype=torch.long, device=device)

    if "bert_mask" not in batch:
        batch["bert_mask"] = torch.zeros_like(batch["true_msa"].float())

    return batch


def make_zero_violation(ref: torch.Tensor) -> dict:
    """
    Build a zero-filled violation dict matching ``find_structural_violations`` output.

    ``AlphaFoldLoss`` unconditionally calls ``find_structural_violations``,
    which requires ``stereo_chemical_props.txt`` — a file that may not be
    present in pip-installed OpenFold builds.  Pre-populating ``raw_out
    ["violation"]`` with this zero dict bypasses the call entirely.

    The violation loss term defaults to weight=0.0, so these zeros have no
    effect on the gradient.

    Args:
        ref: Any tensor from the batch with shape ``[B, L, ...]`` — used only
             to determine device and (B, L) dimensions.

    Returns:
        Dict with the same structure as ``find_structural_violations`` output.
    """
    z1     = ref.new_zeros(())
    B, L   = ref.shape[:2]
    z_BL   = ref.new_zeros(B, L)
    z_BL14 = ref.new_zeros(B, L, 14)
    return {
        "between_residues": {
            "bonds_c_n_loss_mean":                    z1,
            "angles_ca_c_n_loss_mean":                z1,
            "angles_c_n_ca_loss_mean":                z1,
            "connections_per_residue_loss_sum":        z_BL,
            "connections_per_residue_violation_mask":  z_BL,
            "clashes_mean_loss":                       z1,
            "clashes_per_atom_loss_sum":               z_BL14,
            "clashes_per_atom_clash_mask":             z_BL14,
            "clashes_per_atom_num_clash":              z_BL14,
        },
        "within_residues": {
            "per_atom_loss_sum":    z_BL14,
            "per_atom_violations":  z_BL14,
            "per_atom_num_clash":   z_BL14,
        },
        "total_per_residue_violations_mask": z_BL,
    }
