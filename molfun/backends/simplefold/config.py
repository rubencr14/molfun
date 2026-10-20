"""Configuration of the SimpleFold backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimpleFoldConfig:
    """How to build and run a SimpleFold model.

    Attributes:
        architecture: Upstream preset name (e.g. ``"foldingdit_100M"``) or a
            path to a FoldingDiT YAML file.
        esm_model: Frozen protein language model the architecture conditions on.
        multiplicity: Noise copies per structure in each training step.
        num_steps: Sampler steps at inference.
        tau: Sampler diffusion coefficient scaling.
    """

    architecture: str = "foldingdit_100M"
    esm_model: str = "esm2_3B"
    multiplicity: int = 16
    num_steps: int = 500
    tau: float = 0.1
