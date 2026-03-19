"""
Pluggable metric system for model evaluation.

Follows the Observer pattern: metrics accumulate state via ``update()``
and produce final values via ``compute()``.  ``MetricCollection`` composes
any set of metrics behind a single interface (Composite pattern).

All concrete metrics satisfy the Liskov Substitution Principle —
any ``BaseMetric`` subclass can be used wherever a ``BaseMetric`` is expected.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


# ------------------------------------------------------------------
# Abstract interface
# ------------------------------------------------------------------

class BaseMetric(ABC):
    """Interface for a single evaluation metric (Strategy pattern)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short machine-readable identifier (e.g. ``mae``, ``pearson``)."""

    @abstractmethod
    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        """Accumulate a batch of predictions and ground-truth values."""

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Return final metric value(s).  Keys prefixed with ``self.name``."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state for a new evaluation run."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ------------------------------------------------------------------
# Regression metrics
# ------------------------------------------------------------------

class MAE(BaseMetric):
    """Mean Absolute Error."""

    name = "mae"

    def __init__(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        self._sum += (preds.detach().cpu().float() - targets.detach().cpu().float()).abs().sum().item()
        self._count += preds.numel()

    def compute(self) -> dict[str, float]:
        return {self.name: self._sum / max(self._count, 1)}

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0


class RMSE(BaseMetric):
    """Root Mean Squared Error."""

    name = "rmse"

    def __init__(self) -> None:
        self._sum_sq = 0.0
        self._count = 0

    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        diff = preds.detach().cpu().float() - targets.detach().cpu().float()
        self._sum_sq += (diff ** 2).sum().item()
        self._count += preds.numel()

    def compute(self) -> dict[str, float]:
        return {self.name: math.sqrt(self._sum_sq / max(self._count, 1))}

    def reset(self) -> None:
        self._sum_sq = 0.0
        self._count = 0


class PearsonR(BaseMetric):
    """Pearson correlation coefficient (accumulated online)."""

    name = "pearson"

    def __init__(self) -> None:
        self._preds: list[Tensor] = []
        self._targets: list[Tensor] = []

    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        self._preds.append(preds.detach().cpu().float().flatten())
        self._targets.append(targets.detach().cpu().float().flatten())

    def compute(self) -> dict[str, float]:
        if not self._preds:
            return {self.name: 0.0}
        p = torch.cat(self._preds)
        t = torch.cat(self._targets)
        vp = p - p.mean()
        vt = t - t.mean()
        denom = vp.norm() * vt.norm()
        r = (vp * vt).sum() / denom if denom > 1e-8 else torch.tensor(0.0)
        return {self.name: r.item()}

    def reset(self) -> None:
        self._preds.clear()
        self._targets.clear()


class SpearmanRho(BaseMetric):
    """Spearman rank correlation."""

    name = "spearman"

    def __init__(self) -> None:
        self._preds: list[Tensor] = []
        self._targets: list[Tensor] = []

    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        self._preds.append(preds.detach().cpu().float().flatten())
        self._targets.append(targets.detach().cpu().float().flatten())

    def compute(self) -> dict[str, float]:
        if not self._preds:
            return {self.name: 0.0}
        p = torch.cat(self._preds)
        t = torch.cat(self._targets)
        rp = _rank(p)
        rt = _rank(t)
        vp = rp - rp.mean()
        vt = rt - rt.mean()
        denom = vp.norm() * vt.norm()
        rho = (vp * vt).sum() / denom if denom > 1e-8 else torch.tensor(0.0)
        return {self.name: rho.item()}

    def reset(self) -> None:
        self._preds.clear()
        self._targets.clear()


class R2(BaseMetric):
    """Coefficient of determination (R-squared)."""

    name = "r2"

    def __init__(self) -> None:
        self._preds: list[Tensor] = []
        self._targets: list[Tensor] = []

    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        self._preds.append(preds.detach().cpu().float().flatten())
        self._targets.append(targets.detach().cpu().float().flatten())

    def compute(self) -> dict[str, float]:
        if not self._preds:
            return {self.name: 0.0}
        p = torch.cat(self._preds)
        t = torch.cat(self._targets)
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0
        return {self.name: float(r2)}

    def reset(self) -> None:
        self._preds.clear()
        self._targets.clear()


# ------------------------------------------------------------------
# Classification metrics
# ------------------------------------------------------------------

class AUROC(BaseMetric):
    """Area Under Receiver Operating Characteristic (binary)."""

    name = "auroc"

    def __init__(self) -> None:
        self._preds: list[Tensor] = []
        self._targets: list[Tensor] = []

    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        self._preds.append(preds.detach().cpu().float().flatten())
        self._targets.append(targets.detach().cpu().float().flatten())

    def compute(self) -> dict[str, float]:
        if not self._preds:
            return {self.name: 0.0}
        scores = torch.cat(self._preds)
        labels = torch.cat(self._targets)
        return {self.name: _auroc(scores, labels)}

    def reset(self) -> None:
        self._preds.clear()
        self._targets.clear()


class AUPRC(BaseMetric):
    """Area Under Precision-Recall Curve (binary)."""

    name = "auprc"

    def __init__(self) -> None:
        self._preds: list[Tensor] = []
        self._targets: list[Tensor] = []

    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        self._preds.append(preds.detach().cpu().float().flatten())
        self._targets.append(targets.detach().cpu().float().flatten())

    def compute(self) -> dict[str, float]:
        if not self._preds:
            return {self.name: 0.0}
        scores = torch.cat(self._preds)
        labels = torch.cat(self._targets)
        return {self.name: _auprc(scores, labels)}

    def reset(self) -> None:
        self._preds.clear()
        self._targets.clear()


# ------------------------------------------------------------------
# Structural quality metrics
# ------------------------------------------------------------------

class CoordRMSD(BaseMetric):
    """
    Coordinate RMSD between predicted and reference atom positions.

    Expects ``preds`` and ``targets`` of shape ``(N, 3)`` or ``(B, N, 3)``.
    Pass a ``mask`` tensor in ``ctx`` to ignore padded positions.
    """

    name = "coord_rmsd"

    def __init__(self) -> None:
        self._sum_sq = 0.0
        self._count = 0

    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        p = preds.detach().cpu().float()
        t = targets.detach().cpu().float()
        mask = ctx.get("mask")
        diff = (p - t) ** 2
        if p.dim() == 3:
            per_atom = diff.sum(-1)  # (B, N)
            if mask is not None:
                mask = mask.detach().cpu().float()
                per_atom = per_atom * mask
                n = mask.sum().item()
            else:
                n = per_atom.numel()
            self._sum_sq += per_atom.sum().item()
            self._count += int(n)
        else:
            self._sum_sq += diff.sum(-1).sum().item()
            self._count += p.shape[0]

    def compute(self) -> dict[str, float]:
        return {self.name: math.sqrt(self._sum_sq / max(self._count, 1))}

    def reset(self) -> None:
        self._sum_sq = 0.0
        self._count = 0


class GDT_TS(BaseMetric):
    """
    Global Distance Test — Total Score.

    Percentage of C-alpha atoms within {1, 2, 4, 8} Angstrom thresholds,
    averaged.  Standard metric for CASP structure prediction.

    Expects ``preds`` and ``targets`` of shape ``(N, 3)`` or ``(B, N, 3)``.
    """

    name = "gdt_ts"
    _thresholds = (1.0, 2.0, 4.0, 8.0)

    def __init__(self) -> None:
        self._scores: list[float] = []

    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        p = preds.detach().cpu().float()
        t = targets.detach().cpu().float()
        mask = ctx.get("mask")

        if p.dim() == 2:
            p, t = p.unsqueeze(0), t.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        for i in range(p.shape[0]):
            dists = (p[i] - t[i]).norm(dim=-1)  # (N,)
            m = mask[i].bool() if mask is not None else torch.ones(dists.shape, dtype=torch.bool)
            n = m.sum().item()
            if n == 0:
                continue
            fracs = [((dists[m] < th).float().sum().item() / n) for th in self._thresholds]
            self._scores.append(sum(fracs) / len(fracs) * 100.0)

    def compute(self) -> dict[str, float]:
        if not self._scores:
            return {self.name: 0.0}
        return {self.name: sum(self._scores) / len(self._scores)}

    def reset(self) -> None:
        self._scores.clear()


class LDDT(BaseMetric):
    """
    Local Distance Difference Test (lDDT).

    Measures the fraction of inter-residue distances (within a cutoff)
    that are preserved between predicted and reference structures.

    Expects ``preds`` and ``targets`` of shape ``(N, 3)`` or ``(B, N, 3)``.
    """

    name = "lddt"
    _inclusion_radius = 15.0
    _thresholds = (0.5, 1.0, 2.0, 4.0)

    def __init__(self) -> None:
        self._scores: list[float] = []

    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        p = preds.detach().cpu().float()
        t = targets.detach().cpu().float()
        mask = ctx.get("mask")

        if p.dim() == 2:
            p, t = p.unsqueeze(0), t.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        for i in range(p.shape[0]):
            score = _lddt_single(
                p[i], t[i],
                mask=mask[i] if mask is not None else None,
                inclusion_radius=self._inclusion_radius,
                thresholds=self._thresholds,
            )
            if score is not None:
                self._scores.append(score)

    def compute(self) -> dict[str, float]:
        if not self._scores:
            return {self.name: 0.0}
        return {self.name: sum(self._scores) / len(self._scores)}

    def reset(self) -> None:
        self._scores.clear()


class TM_Score(BaseMetric):
    """
    TM-score approximation (length-normalised structural similarity).

    Uses the Zhang & Skolnick (2004) formula with the standard d0
    normalisation.  This is a fast PyTorch approximation — for official
    CASP scoring, use the ``TMscore`` binary.

    Expects ``preds`` and ``targets`` of shape ``(N, 3)`` or ``(B, N, 3)``.
    """

    name = "tm_score"

    def __init__(self) -> None:
        self._scores: list[float] = []

    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        p = preds.detach().cpu().float()
        t = targets.detach().cpu().float()
        mask = ctx.get("mask")

        if p.dim() == 2:
            p, t = p.unsqueeze(0), t.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        for i in range(p.shape[0]):
            m = mask[i].bool() if mask is not None else torch.ones(p.shape[1], dtype=torch.bool)
            pi, ti = p[i][m], t[i][m]
            n = pi.shape[0]
            if n < 5:
                continue
            d0 = 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8
            d0 = max(d0, 0.5)
            dists = (pi - ti).norm(dim=-1)
            tm = (1.0 / (1.0 + (dists / d0) ** 2)).sum().item() / n
            self._scores.append(tm)

    def compute(self) -> dict[str, float]:
        if not self._scores:
            return {self.name: 0.0}
        return {self.name: sum(self._scores) / len(self._scores)}

    def reset(self) -> None:
        self._scores.clear()


class DockingSuccess(BaseMetric):
    """
    Docking success rate: fraction of poses below an RMSD threshold.

    Expects ``preds`` and ``targets`` of shape ``(N, 3)`` (ligand atoms).
    """

    name = "docking_success"

    def __init__(self, threshold: float = 2.0) -> None:
        self._threshold = threshold
        self._hits = 0
        self._total = 0

    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        p = preds.detach().cpu().float()
        t = targets.detach().cpu().float()
        if p.dim() == 2:
            p, t = p.unsqueeze(0), t.unsqueeze(0)
        for i in range(p.shape[0]):
            rmsd = ((p[i] - t[i]) ** 2).sum(-1).mean().sqrt().item()
            self._hits += int(rmsd < self._threshold)
            self._total += 1

    def compute(self) -> dict[str, float]:
        rate = self._hits / max(self._total, 1)
        return {self.name: rate}

    def reset(self) -> None:
        self._hits = 0
        self._total = 0


# ------------------------------------------------------------------
# Composite (collects multiple metrics behind one interface)
# ------------------------------------------------------------------

class MetricCollection:
    """
    Composite of multiple ``BaseMetric`` instances.

    Delegates ``update``/``compute``/``reset`` to all children.
    Implements the Composite pattern.
    """

    def __init__(self, metrics: list[BaseMetric]) -> None:
        self._metrics = list(metrics)

    def update(self, preds: Tensor, targets: Tensor, **ctx) -> None:
        for m in self._metrics:
            m.update(preds, targets, **ctx)

    def compute(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for m in self._metrics:
            result.update(m.compute())
        return result

    def reset(self) -> None:
        for m in self._metrics:
            m.reset()

    def __len__(self) -> int:
        return len(self._metrics)

    def __repr__(self) -> str:
        names = [m.name for m in self._metrics]
        return f"MetricCollection({names})"


# ------------------------------------------------------------------
# Registry: resolve metric by name
# ------------------------------------------------------------------

METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
    "mae": MAE,
    "rmse": RMSE,
    "pearson": PearsonR,
    "spearman": SpearmanRho,
    "r2": R2,
    "auroc": AUROC,
    "auprc": AUPRC,
    "coord_rmsd": CoordRMSD,
    "gdt_ts": GDT_TS,
    "lddt": LDDT,
    "tm_score": TM_Score,
    "docking_success": DockingSuccess,
}


def create_metrics(names: list[str], **kwargs) -> MetricCollection:
    """Factory: build a ``MetricCollection`` from metric name strings."""
    metrics: list[BaseMetric] = []
    for n in names:
        cls = METRIC_REGISTRY.get(n)
        if cls is None:
            raise ValueError(f"Unknown metric: {n!r}. Available: {list(METRIC_REGISTRY)}")
        metrics.append(cls(**kwargs) if kwargs else cls())
    return MetricCollection(metrics)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _rank(x: Tensor) -> Tensor:
    """Assign fractional ranks (average of ties)."""
    order = x.argsort()
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(len(x), dtype=x.dtype)
    return ranks


def _auroc(scores: Tensor, labels: Tensor) -> float:
    """Trapezoidal AUROC for binary labels."""
    if labels.unique().numel() < 2:
        return 0.0
    desc = scores.argsort(descending=True)
    labels = labels[desc].float()
    tpr = labels.cumsum(0) / labels.sum()
    fpr = (1 - labels).cumsum(0) / (1 - labels).sum()
    fpr = torch.cat([torch.tensor([0.0]), fpr])
    tpr = torch.cat([torch.tensor([0.0]), tpr])
    return torch.trapezoid(tpr, fpr).item()


def _auprc(scores: Tensor, labels: Tensor) -> float:
    """Trapezoidal AUPRC for binary labels."""
    if labels.unique().numel() < 2:
        return 0.0
    desc = scores.argsort(descending=True)
    labels = labels[desc].float()
    tp = labels.cumsum(0)
    precision = tp / torch.arange(1, len(labels) + 1, dtype=scores.dtype)
    recall = tp / labels.sum()
    recall = torch.cat([torch.tensor([0.0]), recall])
    precision = torch.cat([torch.tensor([1.0]), precision])
    return torch.trapezoid(precision, recall).abs().item()


def _lddt_single(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor],
    inclusion_radius: float,
    thresholds: tuple[float, ...],
) -> Optional[float]:
    """Compute lDDT for a single structure pair."""
    if mask is not None:
        m = mask.bool()
        pred, target = pred[m], target[m]
    n = pred.shape[0]
    if n < 2:
        return None

    ref_dists = torch.cdist(target.unsqueeze(0), target.unsqueeze(0)).squeeze(0)
    pred_dists = torch.cdist(pred.unsqueeze(0), pred.unsqueeze(0)).squeeze(0)

    within = (ref_dists < inclusion_radius) & ~torch.eye(n, dtype=torch.bool)
    if within.sum() == 0:
        return None

    diff = (ref_dists - pred_dists).abs()
    conserved = sum(((diff[within] < th).float().mean().item()) for th in thresholds)
    return conserved / len(thresholds) * 100.0
