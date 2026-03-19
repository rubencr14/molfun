#!/usr/bin/env python
"""
Build a custom protein structure prediction model from modular components.

This is the key advantage of Molfun for research: you can mix and match
attention mechanisms, trunk blocks, structure modules, and embedders
to rapidly prototype new architectures — no need to rewrite boilerplate.

This example builds an AF3-inspired model:
  - ESM-2 embedder (pre-trained language model features)
  - Pairformer blocks with Flash Attention
  - IPA structure module
  - Affinity prediction head

Then trains it with LoRA on synthetic data to verify everything works.

Requires: pip install molfun
No GPU needed (runs on CPU with small dimensions for prototyping).
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from molfun.modules.builder import ModelBuilder
from molfun.modules import (
    ATTENTION_REGISTRY,
    BLOCK_REGISTRY,
    STRUCTURE_MODULE_REGISTRY,
    EMBEDDER_REGISTRY,
)
from molfun.models.structure import MolfunStructureModel
from molfun.training import LoRAFinetune

# ── 1. Explore available components ──────────────────────────────────

print("Available components:")
print(f"  Attention:  {ATTENTION_REGISTRY.list()}")
print(f"  Blocks:     {BLOCK_REGISTRY.list()}")
print(f"  Structure:  {STRUCTURE_MODULE_REGISTRY.list()}")
print(f"  Embedders:  {EMBEDDER_REGISTRY.list()}")
print()

# ── 2. Build a custom model ──────────────────────────────────────────

# Small dimensions for fast prototyping — scale up for real training
D_SINGLE = 64
D_PAIR = 32

builder = ModelBuilder(
    embedder="input",
    embedder_config={"d_single": D_SINGLE, "d_pair": D_PAIR, "d_msa": D_SINGLE},
    block="pairformer",
    block_config={
        "d_single": D_SINGLE,
        "d_pair": D_PAIR,
        "n_heads": 4,
        "attention_cls": "flash",  # use Flash Attention
    },
    n_blocks=4,
    structure_module="ipa",
    structure_module_config={"d_single": D_SINGLE, "d_pair": D_PAIR},
)

built_model = builder.build()
print(f"Model built: {built_model.param_summary()}")

# ── 3. Wrap as MolfunStructureModel with a head ─────────────────────

model = MolfunStructureModel.from_custom(
    built_model,
    device="cpu",
    head="affinity",
    head_config={"single_dim": D_SINGLE, "hidden_dim": 32},
)

print(f"Full model summary: {model.summary()}")

# ── 4. Quick training test with synthetic data ───────────────────────

SEQ_LEN = 32
N_SAMPLES = 8

def make_synthetic_batch():
    """Minimal batch that matches what the model expects."""
    return {
        "sequences": ["A" * SEQ_LEN],
        "residue_index": torch.arange(SEQ_LEN).unsqueeze(0),
        "all_atom_positions": torch.randn(1, SEQ_LEN, 3),
        "all_atom_mask": torch.ones(1, SEQ_LEN),
        "seq_length": torch.tensor([SEQ_LEN]),
        "aatype": torch.randint(0, 20, (1, SEQ_LEN)),
    }

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        features = {
            "sequence": "A" * SEQ_LEN,
            "residue_index": torch.arange(SEQ_LEN),
            "all_atom_positions": torch.randn(SEQ_LEN, 3),
            "all_atom_mask": torch.ones(SEQ_LEN),
            "seq_length": torch.tensor([SEQ_LEN]),
        }
        label = torch.tensor([5.0 + idx * 0.1])
        return features, label

from molfun.data.datasets.structure import collate_structure_batch

train_loader = DataLoader(SyntheticDataset(6), batch_size=1, collate_fn=collate_structure_batch)
val_loader = DataLoader(SyntheticDataset(2), batch_size=1, collate_fn=collate_structure_batch)

strategy = LoRAFinetune(
    rank=4,
    lr_lora=1e-3,
    lr_head=1e-2,
    warmup_steps=0,
    loss_fn="mse",
)

print("\nTraining for 3 epochs...")
history = strategy.fit(model, train_loader, val_loader, epochs=3)

for epoch_info in history:
    print(f"  Epoch {epoch_info.get('epoch', '?')}: "
          f"train_loss={epoch_info.get('train_loss', 0):.4f}, "
          f"val_loss={epoch_info.get('val_loss', 0):.4f}")

# ── 5. Experiment with different architectures ───────────────────────

print("\n--- Trying a different architecture ---")

builder2 = ModelBuilder(
    embedder="input",
    embedder_config={"d_single": D_SINGLE, "d_pair": D_PAIR, "d_msa": D_SINGLE},
    block="simple_transformer",
    block_config={
        "d_single": D_SINGLE,
        "n_heads": 4,
        "attention_cls": "gated",  # gated attention (learnable sigmoid gate)
    },
    n_blocks=6,
    structure_module="diffusion",
    structure_module_config={"d_single": D_SINGLE, "d_pair": D_PAIR},
)

built2 = builder2.build()
print(f"Alternative model: {built2.param_summary()}")
print("Swapping components is this easy — no boilerplate, just change the config.")
