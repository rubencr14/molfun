"""Parameter-free head for generative backends.

Generative structure models (flow matching, diffusion) compute their own
training losses inside the adapter, because the loss depends on the noise
and timestep they sample. The adapter reports them in
``TrunkOutput.extra["losses"]``; this head hands them to the training loop,
which optimises the first entry (``"total"``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GenerativeLossHead(nn.Module):
    """Forward the adapter's loss dict to the training loop."""

    def forward(self, trunk_output, mask=None, batch=None) -> dict[str, torch.Tensor]:
        """Return ``trunk_output.extra["losses"]``.

        Raises:
            KeyError: If the adapter did not report losses (e.g. it ran in
                eval mode or is not a generative backend).
        """
        losses = trunk_output.extra.get("losses")
        if losses is None:
            raise KeyError(
                "TrunkOutput.extra has no 'losses'; the generative head needs an "
                "adapter that computes its losses in training mode"
            )
        return losses

    def loss(self, preds: dict, targets=None, loss_fn=None) -> dict[str, torch.Tensor]:
        """Return the loss dict with ``"total"`` first (the loop optimises the first entry)."""
        if "total" not in preds:
            raise KeyError("generative losses must include a 'total' entry")
        return {"total": preds["total"], **{k: v for k, v in preds.items() if k != "total"}}
