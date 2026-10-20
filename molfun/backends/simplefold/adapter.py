"""SimpleFold (flow-matching folding transformer) as a molfun backend.

The adapter wraps upstream's ``SimpleFold`` LightningModule but drives it
without a Lightning Trainer: molfun's training loop calls ``forward(batch)``,
which runs upstream's own ``flow_matching_train_step`` (ESM features, noise,
timestep sampling, velocity MSE + smooth-lDDT) and reports the losses through
``TrunkOutput.extra["losses"]`` for the ``generative`` head.

Only the FoldingDiT architecture is trainable; the ESM language model is
always frozen and kept in eval mode. Upstream's internal EMA copy is dropped:
molfun's strategies keep their own EMA.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from molfun.adapters.base import BaseAdapter
from molfun.backends.simplefold.config import SimpleFoldConfig
from molfun.backends.simplefold.imports import import_simplefold
from molfun.core.types import TrunkOutput

_LOG_TO_LOSS = {"loss/mse": "flow_mse", "loss/smooth_lddt": "smooth_lddt"}


class _StandaloneTrainer:
    """The few Trainer attributes upstream's training step reads."""

    global_step = 0
    current_epoch = 0
    world_size = 1


def build_architecture(spec: str) -> nn.Module:
    """Instantiate a FoldingDiT from an upstream preset name or a YAML path."""
    import hydra
    import omegaconf

    sf = import_simplefold()
    path = Path(spec)
    if not path.is_file():
        path = sf.architecture_dir / f"{spec}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No SimpleFold architecture {spec!r} (looked for {path})")
    return hydra.utils.instantiate(omegaconf.OmegaConf.load(path))


class SimpleFoldAdapter(BaseAdapter):
    """molfun adapter for SimpleFold."""

    def __init__(
        self,
        model: nn.Module | None = None,
        config: SimpleFoldConfig | None = None,
        weights_path: str | None = None,
        device: str = "cuda",
    ):
        """
        Args:
            model: A pre-built FoldingDiT architecture (built from ``config`` if None).
            config: Backend configuration.
            weights_path: Released checkpoint (a FoldingDiT state dict), loaded strictly.
            device: Target device.
        """
        super().__init__()
        sf = import_simplefold()
        self.config = config or SimpleFoldConfig()
        self._device = torch.device(device)

        architecture = model if model is not None else build_architecture(self.config.architecture)
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            architecture.load_state_dict(state, strict=True)

        processor = sf.ProteinDataProcessor(
            device=self._device, scale=16.0, ref_scale=5.0, multiplicity=self.config.multiplicity
        )
        self.sf = sf.SimpleFold(
            architecture=architecture,
            processor=processor,
            loss=nn.MSELoss(reduction="none"),
            path=sf.LinearPath(),
            sampler=sf.EMSampler(
                num_timesteps=self.config.num_steps,
                t_start=1e-4,
                tau=self.config.tau,
                log_timesteps=True,
                w_cutoff=0.99,
            ),
            esm_model=self.config.esm_model,
        )
        # What upstream's Lightning ``setup()`` hook would do (non-FSDP branch).
        self.sf.processor = processor
        self.sf.model_ema = None
        self.sf._trainer = _StandaloneTrainer()
        self.sf.log = self._capture_log
        self._logged: dict[str, float] = {}

        for p in self.sf.esm_model.parameters():
            p.requires_grad = False
        self.sf.to(self._device)
        self.sf.af2_to_esm = self.sf.af2_to_esm.to(self._device)
        self.sf.esm_model.eval()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _capture_log(self, name: str, value, *args, **kwargs) -> None:
        self._logged[name] = float(value)

    def forward(self, batch: dict) -> TrunkOutput:
        """Compute SimpleFold's training losses on one collated batch."""
        self._logged = {}
        total = self.sf.flow_matching_train_step(batch, 0)
        losses = {"total": total}
        for log_name, loss_name in _LOG_TO_LOSS.items():
            value = self._logged.get(log_name, 0.0 if loss_name == "smooth_lddt" else math.nan)
            losses[loss_name] = torch.tensor(value, device=total.device)
        return TrunkOutput(extra={"losses": losses})

    def train(self, mode: bool = True):
        super().train(mode)
        self.sf.esm_model.eval()
        return self

    # ------------------------------------------------------------------
    # BaseAdapter contract
    # ------------------------------------------------------------------

    def freeze_trunk(self) -> None:
        for p in self.sf.model.parameters():
            p.requires_grad = False

    def unfreeze_trunk(self) -> None:
        for p in self.sf.model.parameters():
            p.requires_grad = True

    def get_trunk_blocks(self) -> nn.ModuleList:
        """The residue-level transformer blocks (not the atom encoder/decoder)."""
        return self.sf.model.trunk.blocks

    @property
    def peft_target_module(self) -> nn.Module:
        return self.sf.model

    @property
    def default_peft_targets(self) -> list[str]:
        return ["qkv", "proj", "w1", "w2"]

    def param_summary(self) -> dict[str, int]:
        trunk = sum(p.numel() for p in self.sf.model.parameters())
        return {
            "trunk": trunk,
            "trainable": sum(p.numel() for p in self.sf.model.parameters() if p.requires_grad),
            "esm_frozen": sum(p.numel() for p in self.sf.esm_model.parameters()),
        }
