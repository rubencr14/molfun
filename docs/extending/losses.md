# Adding Loss Functions

Loss functions in Molfun return a dictionary of named scalar tensors, enabling callers to log individual loss terms without knowing the internals. The `LOSS_REGISTRY` provides name-based lookup so training strategies and heads can resolve losses at runtime.

## The LossFunction interface

All losses inherit from `LossFunction` in `molfun/losses/base.py`:

```python
class LossFunction(ABC, nn.Module):
    """
    Abstract base for all Molfun loss functions.

    Signature: loss_fn(preds, targets=None, batch=None) -> dict[str, Tensor]
    """

    @abstractmethod
    def forward(
        self,
        preds: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        batch: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss and return a dict of named scalar tensors."""
```

Key design points:

- **Returns a dict**, not a single tensor. This allows composite losses to expose individual terms for logging.
- `targets` is optional because some losses extract ground truth from the `batch` dict directly (e.g., structure losses that need atom coordinates from the feature dict).
- `batch` is the raw feature dict from the DataLoader, forwarded for losses that need additional context.

## The LossRegistry

The `LossRegistry` is separate from `ModuleRegistry` because loss functions have a simpler API:

```python
LOSS_REGISTRY.register("name")   # decorator
LOSS_REGISTRY["name"]            # get class (raises KeyError if missing)
LOSS_REGISTRY["name"]()          # instantiate with defaults
"name" in LOSS_REGISTRY          # membership test
list(LOSS_REGISTRY)              # all registered names
```

## Example: Contact Map Loss

A contact map loss penalizes predicted structures where residue pairs that should be in contact (based on ground-truth distance maps) are too far apart, and vice versa.

### Step 1: Create the loss file

Create `molfun/losses/contact.py`:

```python
"""
Contact map loss: penalizes incorrect inter-residue distance predictions.

Useful as an auxiliary loss alongside FAPE/coordinate losses, providing
a complementary signal about the global topology of the predicted structure.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from molfun.losses.base import LossFunction, LOSS_REGISTRY


@LOSS_REGISTRY.register("contact")
class ContactMapLoss(LossFunction):
    """
    Binary cross-entropy loss on predicted vs. ground-truth contact maps.

    A contact is defined as two C-alpha atoms within ``threshold`` angstroms.
    The loss operates on predicted positions and ground-truth positions,
    both expected in the ``batch`` dict.

    Args:
        threshold: Distance threshold in angstroms for defining a contact.
        weight: Scalar weight for the loss term.
        min_seq_sep: Minimum sequence separation to consider (ignore trivial
            contacts between adjacent residues).
    """

    def __init__(
        self,
        threshold: float = 8.0,
        weight: float = 1.0,
        min_seq_sep: int = 6,
    ):
        super().__init__()
        self.threshold = threshold
        self.weight = weight
        self.min_seq_sep = min_seq_sep

    def _compute_contact_map(
        self, positions: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute binary contact map from C-alpha positions.

        Args:
            positions: [B, L, 3] C-alpha coordinates.
            mask: [B, L] residue mask.

        Returns:
            [B, L, L] binary contact map (1 = contact, 0 = no contact).
        """
        # Pairwise distances
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, L, L, 3]
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-8)           # [B, L, L]

        contacts = (dist < self.threshold).float()

        # Zero out contacts below minimum sequence separation
        L = positions.shape[1]
        seq_idx = torch.arange(L, device=positions.device)
        sep = (seq_idx.unsqueeze(1) - seq_idx.unsqueeze(0)).abs()
        sep_mask = (sep >= self.min_seq_sep).float()
        contacts = contacts * sep_mask.unsqueeze(0)

        if mask is not None:
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            contacts = contacts * pair_mask

        return contacts

    def forward(
        self,
        preds: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        batch: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            preds: Predicted positions [B, L, 3].
            targets: Ground-truth positions [B, L, 3]. If None, extracted
                from ``batch["gt_positions"]``.
            batch: Feature dict from the DataLoader.

        Returns:
            Dict with ``"contact_loss"`` key.
        """
        if targets is None and batch is not None:
            targets = batch["gt_positions"]
        if targets is None:
            raise ValueError("ContactMapLoss requires targets or batch['gt_positions']")

        mask = batch.get("mask") if batch is not None else None

        pred_contacts = self._compute_contact_map(preds, mask)
        true_contacts = self._compute_contact_map(targets, mask)

        # Binary cross-entropy (with logits for numerical stability)
        # Convert pred distances to logits
        pred_diff = preds.unsqueeze(2) - preds.unsqueeze(1)
        pred_dist = torch.sqrt((pred_diff ** 2).sum(-1) + 1e-8)
        pred_logits = self.threshold - pred_dist  # positive = contact

        loss = F.binary_cross_entropy_with_logits(
            pred_logits, true_contacts, reduction="none"
        )

        # Apply sequence separation mask
        L = preds.shape[1]
        seq_idx = torch.arange(L, device=preds.device)
        sep = (seq_idx.unsqueeze(1) - seq_idx.unsqueeze(0)).abs()
        sep_mask = (sep >= self.min_seq_sep).float().unsqueeze(0)
        loss = loss * sep_mask

        if mask is not None:
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            loss = loss * pair_mask
            loss = loss.sum() / pair_mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return {"contact_loss": self.weight * loss}
```

### Step 2: Register via __init__.py

Add the import to `molfun/losses/__init__.py`:

```python
from molfun.losses.contact import ContactMapLoss  # noqa: F401
```

## Testing

Create `tests/losses/test_contact.py`:

```python
import pytest
import torch

from molfun.losses.base import LOSS_REGISTRY


class TestContactMapLoss:

    @pytest.fixture
    def loss_fn(self):
        return LOSS_REGISTRY["contact"](threshold=8.0, min_seq_sep=3)

    def test_registry_lookup(self):
        assert "contact" in LOSS_REGISTRY

    def test_returns_dict(self, loss_fn):
        preds = torch.randn(2, 10, 3)
        targets = torch.randn(2, 10, 3)
        result = loss_fn(preds, targets)
        assert isinstance(result, dict)
        assert "contact_loss" in result

    def test_scalar_output(self, loss_fn):
        preds = torch.randn(2, 10, 3)
        targets = torch.randn(2, 10, 3)
        result = loss_fn(preds, targets)
        assert result["contact_loss"].dim() == 0  # scalar

    def test_perfect_prediction(self, loss_fn):
        """Loss should be low when prediction matches target."""
        targets = torch.randn(2, 20, 3)
        # Identical predictions
        result_same = loss_fn(targets, targets)
        # Wildly different predictions
        result_diff = loss_fn(targets * 10, targets)
        # Same should have lower loss
        assert result_same["contact_loss"] < result_diff["contact_loss"]

    def test_with_batch_dict(self, loss_fn):
        """Can extract targets from batch dict."""
        preds = torch.randn(2, 10, 3)
        batch = {"gt_positions": torch.randn(2, 10, 3)}
        result = loss_fn(preds, batch=batch)
        assert "contact_loss" in result

    def test_with_mask(self, loss_fn):
        preds = torch.randn(2, 10, 3)
        targets = torch.randn(2, 10, 3)
        mask = torch.ones(2, 10)
        mask[:, -3:] = 0
        batch = {"mask": mask}
        result = loss_fn(preds, targets, batch=batch)
        assert "contact_loss" in result

    def test_gradient_flow(self, loss_fn):
        preds = torch.randn(2, 10, 3, requires_grad=True)
        targets = torch.randn(2, 10, 3)
        result = loss_fn(preds, targets)
        result["contact_loss"].backward()
        assert preds.grad is not None

    def test_weight_scaling(self):
        """Weight parameter scales the loss."""
        preds = torch.randn(2, 10, 3)
        targets = torch.randn(2, 10, 3)
        loss_w1 = LOSS_REGISTRY["contact"](weight=1.0)(preds, targets)
        loss_w5 = LOSS_REGISTRY["contact"](weight=5.0)(preds, targets)
        torch.testing.assert_close(
            loss_w5["contact_loss"],
            loss_w1["contact_loss"] * 5.0,
            rtol=1e-5, atol=1e-5,
        )
```

## Integration: Using in training

### As the primary loss

```python
from molfun import MolfunStructureModel
from molfun.training import HeadOnlyFinetune

strategy = HeadOnlyFinetune(lr=1e-3, loss_fn="contact")
model = MolfunStructureModel("openfold")
strategy.fit(model, train_loader, val_loader, epochs=20)
```

### In combination with other losses

Since losses return dicts, you can compose them:

```python
from molfun.losses.base import LOSS_REGISTRY

fape_fn = LOSS_REGISTRY["fape"]()
contact_fn = LOSS_REGISTRY["contact"](weight=0.5)

# In a custom training loop:
fape_losses = fape_fn(preds, targets, batch)
contact_losses = contact_fn(preds.positions, batch=batch)

all_losses = {**fape_losses, **contact_losses}
total = sum(all_losses.values())
```
