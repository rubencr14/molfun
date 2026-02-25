"""
Shared fixtures for integration tests.

All fixtures use small dimensions (D=32, L=8) so tests run in seconds on CPU.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

D_SINGLE = 32
D_PAIR = 16
SEQ_LEN = 8
N_SAMPLES = 12
BATCH_SIZE = 4
DEVICE = "cpu"


class SyntheticAffinityDataset(Dataset):
    """Produces (dict_batch, target) pairs that BuiltModel can consume."""

    def __init__(self, n: int = N_SAMPLES, seq_len: int = SEQ_LEN):
        self.n = n
        self.seq_len = seq_len
        self.targets = torch.randn(n, 1)
        self.aatype = torch.randint(0, 20, (n, seq_len))
        self.residue_index = torch.arange(seq_len).unsqueeze(0).expand(n, -1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        feats = {
            "aatype": self.aatype[idx],
            "residue_index": self.residue_index[idx],
        }
        return feats, self.targets[idx]


def make_loader(n: int = N_SAMPLES, batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN) -> DataLoader:
    return DataLoader(SyntheticAffinityDataset(n, seq_len), batch_size=batch_size)


def build_custom_model(
    block: str = "pairformer",
    block_config: dict | None = None,
    n_blocks: int = 2,
) -> "MolfunStructureModel":
    from molfun.modules.builder import ModelBuilder
    from molfun.models.structure import MolfunStructureModel

    default_block_config = {
        "d_single": D_SINGLE,
        "d_pair": D_PAIR,
        "n_heads": 4,
        "attention_cls": "standard",
    }
    if block_config:
        default_block_config.update(block_config)

    built = ModelBuilder(
        embedder="input",
        embedder_config={"d_single": D_SINGLE, "d_pair": D_PAIR, "d_msa": D_SINGLE},
        block=block,
        block_config=default_block_config,
        n_blocks=n_blocks,
        structure_module="ipa",
        structure_module_config={"d_single": D_SINGLE, "d_pair": D_PAIR},
    ).build()
    return MolfunStructureModel.from_custom(
        built, device=DEVICE, head="affinity",
        head_config={"single_dim": D_SINGLE, "hidden_dim": 16},
    )
