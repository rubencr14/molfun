"""Base adapter interface that all model adapters must implement."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn

from molfun.core.types import TrunkOutput


class BaseAdapter(nn.Module, ABC):
    """
    Normalized interface for structure prediction models.

    Every adapter must expose:
    - forward(batch) → TrunkOutput
    - freeze_trunk() / unfreeze_trunk()
    - get_trunk_blocks() → nn.ModuleList (the main repeating blocks)
    - peft_target_module → nn.Module (the submodule where PEFT is injected)
    - param_summary() → dict
    """

    @abstractmethod
    def forward(self, batch: dict) -> TrunkOutput:
        ...

    @abstractmethod
    def freeze_trunk(self) -> None:
        ...

    @abstractmethod
    def unfreeze_trunk(self) -> None:
        ...

    @abstractmethod
    def get_trunk_blocks(self) -> nn.ModuleList:
        """
        Return the main repeating blocks of the model trunk.

        These are the units that PartialFinetune unfreezes from the end
        and FullFinetune assigns layer-wise LR decay to.

        For OpenFold this is the Evoformer blocks, for Protenix/Boltz the
        Pairformer/trunk blocks, for ESMFold the transformer layers, etc.
        """
        ...

    @property
    @abstractmethod
    def peft_target_module(self) -> nn.Module:
        """Return the submodule where PEFT layers should be injected."""
        ...

    @abstractmethod
    def param_summary(self) -> dict[str, int]:
        ...

    # ------------------------------------------------------------------
    # Optional overrides — sensible defaults for simple adapters
    # ------------------------------------------------------------------

    def get_structure_module(self) -> Optional[nn.Module]:
        """
        Return the structure prediction module (IPA, diffusion, etc.).

        Used by PartialFinetune to optionally unfreeze it, and by
        FullFinetune to assign a separate learning rate.

        Returns None if the model doesn't have a distinct structure module
        (e.g. ESMFold where folding is part of the trunk).
        """
        return None

    def get_input_embedder(self) -> Optional[nn.Module]:
        """
        Return the input embedding module.

        Used by FullFinetune to assign the lowest learning rate
        (embeddings change the least during fine-tuning).

        Returns None if there is no distinct embedder.
        """
        return None

    @property
    def default_peft_targets(self) -> list[str]:
        """
        Return the default layer name substrings for PEFT injection.

        LoRAFinetune uses these when the user doesn't specify
        ``target_modules`` explicitly. Each adapter knows the naming
        convention of its own attention projections.

        Examples:
            OpenFold:  ["linear_q", "linear_v"]
            Protenix:  ["q_proj", "v_proj"]
            ESMFold:   ["q_proj", "v_proj"]
        """
        return []
