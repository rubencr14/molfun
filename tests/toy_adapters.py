"""Small adapters for tests that need a real (not mocked) molfun backend."""

import torch.nn as nn

from molfun.adapters.base import BaseAdapter
from molfun.core.types import TrunkOutput


class ToyGenerativeAdapter(BaseAdapter):
    """Two linear blocks; loss = mse term + abs term, reported in extra."""

    def __init__(self, model=None, config=None, weights_path=None, device="cpu"):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.to(device)

    def forward(self, batch):
        x = batch["x"]
        for block in self.get_trunk_blocks():
            x = block(x)
        flow = (x.float() ** 2).mean()
        lddt = x.float().abs().mean()
        return TrunkOutput(
            extra={"losses": {"total": flow + lddt, "flow_mse": flow, "smooth_lddt": lddt}}
        )

    def freeze_trunk(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_trunk(self):
        for p in self.parameters():
            p.requires_grad = True

    def get_trunk_blocks(self):
        return self.blocks

    @property
    def peft_target_module(self):
        return self

    def param_summary(self):
        return {"total": sum(p.numel() for p in self.parameters())}


class RenamedToyGenerativeAdapter(ToyGenerativeAdapter):
    """Same shapes as ToyGenerativeAdapter, different parameter names."""

    def __init__(self, model=None, config=None, weights_path=None, device="cpu"):
        BaseAdapter.__init__(self)
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.to(device)

    def get_trunk_blocks(self):
        return self.layers
