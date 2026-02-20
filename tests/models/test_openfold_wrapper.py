"""
End-to-end test for structure models: adapter → PEFT → head → train loop.
Uses a mock model so OpenFold installation is not required.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from molfun.models.structure import MolfunStructureModel
from molfun.models.openfold import OpenFold
from molfun.adapters.openfold import OpenFoldAdapter
from molfun.peft.lora import MolfunPEFT, LoRALinear
from molfun.heads.affinity import AffinityHead
from molfun.core.types import TrunkOutput


# ── Mock OpenFold model ───────────────────────────────────────────────

class _MockEvoformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.ff_linear1 = nn.Linear(dim, dim * 4)
        self.ff_linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        attn = torch.softmax(q @ k.transpose(-1, -2), dim=-1) @ v
        return self.ff_linear2(torch.relu(self.ff_linear1(attn)))


class _MockEvoformer(nn.Module):
    def __init__(self, dim: int, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_MockEvoformerBlock(dim) for _ in range(n_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class _MockOpenFold(nn.Module):
    """Minimal mock replicating the OpenFold interface needed by the adapter."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.evoformer = _MockEvoformer(dim, n_blocks=2)
        self.structure_module = nn.Linear(dim, dim)
        self.input_embedder = nn.Linear(dim, dim)
        self._dim = dim

    def forward(self, batch):
        B, L = 2, 10
        x = torch.randn(B, L, self._dim, device=next(self.parameters()).device)
        single = self.evoformer(x)
        return {
            "single": single,
            "pair": torch.randn(B, L, L, self._dim, device=single.device),
            "final_atom_positions": torch.randn(B, L, 3, device=single.device),
            "plddt": torch.rand(B, L, device=single.device),
        }


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 64


# ── Tests ─────────────────────────────────────────────────────────────

class TestOpenFoldWrapper:

    def _make_model(self, **kw) -> OpenFold:
        mock = _MockOpenFold(DIM)
        defaults = dict(model=mock, device=DEVICE)
        defaults.update(kw)
        return OpenFold(**defaults)

    def test_predict_inference(self):
        wrapper = self._make_model()
        out = wrapper.predict({})
        assert isinstance(out, TrunkOutput)
        assert out.single_repr.shape[-1] == DIM

    def test_fine_tune_lora_builtin(self):
        wrapper = self._make_model(
            fine_tune=True,
            peft="lora",
            peft_config={"rank": 4, "alpha": 8.0, "target_modules": ["linear_q", "linear_v"], "use_hf": False},
            head="affinity",
            head_config={"single_dim": DIM, "hidden_dim": 32},
        )
        info = wrapper.summary()
        assert info["fine_tune"] is True
        assert info["peft"]["method"] == "lora"
        assert info["head"]["type"] == "AffinityHead"

        result = wrapper.forward({})
        assert "preds" in result
        assert result["preds"].shape[-1] == 1

    def test_fit_one_epoch(self):
        wrapper = self._make_model(
            fine_tune=True,
            peft="lora",
            peft_config={"rank": 4, "target_modules": ["linear_q", "linear_v"], "use_hf": False},
            head="affinity",
            head_config={"single_dim": DIM, "hidden_dim": 32},
        )

        N = 4
        dummy_features = [{} for _ in range(N)]
        dummy_targets = torch.randn(N, 1, device=DEVICE)

        dataset = list(zip(dummy_features, dummy_targets))
        loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)

        history = wrapper.fit(loader, epochs=1, lr=1e-3, amp=False)
        assert len(history) == 1
        assert "train_loss" in history[0]

    def test_save_load(self, tmp_path):
        wrapper = self._make_model(
            fine_tune=True,
            peft="lora",
            peft_config={"rank": 4, "target_modules": ["linear_q", "linear_v"], "use_hf": False},
            head="affinity",
            head_config={"single_dim": DIM, "hidden_dim": 32},
        )

        save_dir = str(tmp_path / "checkpoint")
        wrapper.save(save_dir)

        wrapper2 = self._make_model(
            fine_tune=True,
            peft="lora",
            peft_config={"rank": 4, "target_modules": ["linear_q", "linear_v"], "use_hf": False},
            head="affinity",
            head_config={"single_dim": DIM, "hidden_dim": 32},
        )
        wrapper2.load(save_dir)

    def test_merge_unmerge(self):
        wrapper = self._make_model(
            fine_tune=True,
            peft="lora",
            peft_config={"rank": 4, "target_modules": ["linear_q", "linear_v"], "use_hf": False},
            head="affinity",
            head_config={"single_dim": DIM, "hidden_dim": 32},
        )

        out_before = wrapper.forward({})
        wrapper.merge()
        out_merged = wrapper.forward({})
        wrapper.unmerge()

        assert out_before["preds"].shape == out_merged["preds"].shape

    def test_head_only_no_peft(self):
        wrapper = self._make_model(
            fine_tune=True,
            head="affinity",
            head_config={"single_dim": DIM, "hidden_dim": 32},
        )
        info = wrapper.summary()
        assert "peft" not in info
        assert info["head"]["type"] == "AffinityHead"

        trunk_trainable = info["adapter"]["trainable"]
        assert trunk_trainable == 0


class TestAdapterStandalone:

    def test_targetable_modules(self):
        mock = _MockOpenFold(DIM).to(DEVICE)
        adapter = OpenFoldAdapter(model=mock, device=DEVICE)
        mods = adapter.get_targetable_modules()
        assert "evoformer" in mods
        assert "evoformer.blocks.0" in mods

    def test_freeze_unfreeze(self):
        mock = _MockOpenFold(DIM).to(DEVICE)
        adapter = OpenFoldAdapter(model=mock, device=DEVICE)
        adapter.freeze_trunk()
        assert all(not p.requires_grad for p in adapter.model.parameters())
        adapter.unfreeze_trunk()
        assert all(p.requires_grad for p in adapter.model.parameters())


class TestPEFTStandalone:

    def test_builtin_lora_injection(self):
        mock = _MockOpenFold(DIM).to(DEVICE)
        peft = MolfunPEFT.lora(rank=4, target_modules=["linear_q", "linear_v"], use_hf=False)
        peft.apply(mock.evoformer)

        for block in mock.evoformer.blocks:
            assert isinstance(block.linear_q, LoRALinear)
            assert isinstance(block.linear_v, LoRALinear)
            assert isinstance(block.linear_k, nn.Linear) and not isinstance(block.linear_k, LoRALinear)

    def test_trainable_param_count(self):
        mock = _MockOpenFold(DIM).to(DEVICE)
        peft = MolfunPEFT.lora(rank=4, target_modules=["linear_q"], use_hf=False)
        peft.apply(mock.evoformer)
        count = peft.trainable_param_count()
        expected = 2 * (4 * DIM + DIM * 4)  # 2 blocks × (lora_A + lora_B)
        assert count == expected


class TestAffinityHeadStandalone:

    def test_forward_shape(self):
        head = AffinityHead(single_dim=DIM, hidden_dim=32).to(DEVICE)
        trunk = TrunkOutput(
            single_repr=torch.randn(2, 10, DIM, device=DEVICE),
            pair_repr=None,
            structure_coords=None,
            confidence=None,
        )
        out = head(trunk)
        assert out.shape == (2, 1)

    def test_attention_pool(self):
        head = AffinityHead(single_dim=DIM, hidden_dim=32, pool="attention").to(DEVICE)
        trunk = TrunkOutput(
            single_repr=torch.randn(2, 10, DIM, device=DEVICE),
            pair_repr=None,
            structure_coords=None,
            confidence=None,
        )
        out = head(trunk)
        assert out.shape == (2, 1)

    def test_loss(self):
        head = AffinityHead(single_dim=DIM, hidden_dim=32).to(DEVICE)
        preds = torch.randn(4, 1, device=DEVICE, requires_grad=True)
        targets = torch.randn(4, device=DEVICE)
        losses = head.loss(preds, targets)
        assert "affinity_loss" in losses
        assert losses["affinity_loss"].requires_grad


# ── MolfunStructureModel (unified API) ────────────────────────────────

class TestMolfunStructureModel:

    def _make(self, **kw) -> MolfunStructureModel:
        mock = _MockOpenFold(DIM)
        defaults = dict(name="openfold", model=mock, device=DEVICE)
        defaults.update(kw)
        return MolfunStructureModel(**defaults)

    def test_predict(self):
        m = self._make()
        out = m.predict({})
        assert isinstance(out, TrunkOutput)

    def test_fine_tune_lora(self):
        m = self._make(
            fine_tune=True,
            peft="lora",
            peft_config={"rank": 4, "target_modules": ["linear_q", "linear_v"], "use_hf": False},
            head="affinity",
            head_config={"single_dim": DIM, "hidden_dim": 32},
        )
        info = m.summary()
        assert info["name"] == "openfold"
        assert info["peft"]["method"] == "lora"
        assert info["head"]["type"] == "AffinityHead"

    def test_fit(self):
        m = self._make(
            fine_tune=True,
            peft="lora",
            peft_config={"rank": 4, "target_modules": ["linear_q", "linear_v"], "use_hf": False},
            head="affinity",
            head_config={"single_dim": DIM, "hidden_dim": 32},
        )
        dataset = list(zip([{} for _ in range(4)], torch.randn(4, 1, device=DEVICE)))
        loader = DataLoader(dataset, batch_size=2, collate_fn=_collate_fn)
        history = m.fit(loader, epochs=1, lr=1e-3, amp=False)
        assert len(history) == 1

    def test_save_load(self, tmp_path):
        m = self._make(
            fine_tune=True,
            peft="lora",
            peft_config={"rank": 4, "target_modules": ["linear_q"], "use_hf": False},
            head="affinity",
            head_config={"single_dim": DIM, "hidden_dim": 32},
        )
        m.save(str(tmp_path / "ckpt"))
        assert (tmp_path / "ckpt" / "meta.pt").exists()
        assert (tmp_path / "ckpt" / "head.pt").exists()

        m2 = self._make(
            fine_tune=True,
            peft="lora",
            peft_config={"rank": 4, "target_modules": ["linear_q"], "use_hf": False},
            head="affinity",
            head_config={"single_dim": DIM, "hidden_dim": 32},
        )
        m2.load(str(tmp_path / "ckpt"))

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            MolfunStructureModel("nonexistent", config={})

    def test_unknown_head_raises(self):
        mock = _MockOpenFold(DIM)
        with pytest.raises(ValueError, match="Unknown head"):
            MolfunStructureModel("openfold", model=mock, device=DEVICE, fine_tune=True, head="bad")

    def test_unknown_peft_raises(self):
        mock = _MockOpenFold(DIM)
        with pytest.raises(ValueError, match="Unknown PEFT"):
            MolfunStructureModel("openfold", model=mock, device=DEVICE, fine_tune=True, peft="bad")

    def test_available_models(self):
        models = MolfunStructureModel.available_models()
        assert "openfold" in models

    def test_available_heads(self):
        heads = MolfunStructureModel.available_heads()
        assert "affinity" in heads

    def test_openfold_alias_is_subclass(self):
        m = OpenFold(model=_MockOpenFold(DIM), device=DEVICE)
        assert isinstance(m, MolfunStructureModel)


# ── Helpers ───────────────────────────────────────────────────────────

def _collate_fn(batch):
    features, targets = zip(*batch)
    return list(features)[0], torch.stack(targets), None
