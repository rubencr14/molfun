"""
Real OpenFold fine-tuning test on GPU.

Requires:
    - OpenFold installed: pip install git+https://github.com/aqlaboratory/openfold.git
    - Weights at: ~/.molfun/weights/finetuning_ptm_2.pt
    - GPU with >= 16GB VRAM

Run:
    pytest tests/models/test_openfold_real.py -v -s
"""

import pytest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

WEIGHTS_PATH = Path.home() / ".molfun" / "weights" / "finetuning_ptm_2.pt"


def _openfold_available() -> bool:
    try:
        from openfold.model.model import AlphaFold
        return True
    except ImportError:
        return False


skip_no_openfold = pytest.mark.skipif(
    not _openfold_available(),
    reason="OpenFold not installed",
)
skip_no_weights = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason=f"Weights not found at {WEIGHTS_PATH}",
)
skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


# ── Dummy dataset that produces OpenFold-compatible features ──────────

class DummyOpenFoldDataset(Dataset):
    """
    Generates synthetic OpenFold-format features for testing.

    Uses OpenFold's own data_transforms to build atom14/atom37 features,
    then appends a trailing recycling dimension (size R=1).
    """

    def __init__(self, n_samples: int = 4, seq_len: int = 32, n_msa: int = 8):
        self.n = n_samples
        self.L = seq_len
        self.n_msa = n_msa

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        from openfold.data.data_transforms import make_atom14_masks, make_atom14_positions

        L, N = self.L, self.n_msa
        T = 1   # templates

        aatype = torch.randint(0, 20, (L,))

        protein = {
            "aatype": aatype,
            "all_atom_positions": torch.zeros(L, 37, 3),
            "all_atom_mask": torch.zeros(L, 37),
        }
        protein = make_atom14_masks(protein)
        protein = make_atom14_positions(protein)

        features = {
            "aatype": aatype,
            "residue_index": torch.arange(L),
            "msa_feat": torch.randn(N, L, 49),
            "target_feat": torch.randn(L, 22),
            "extra_msa": torch.randint(0, 23, (N * 4, L)),
            "extra_has_deletion": torch.zeros(N * 4, L),
            "extra_deletion_value": torch.zeros(N * 4, L),
            "extra_msa_mask": torch.ones(N * 4, L),
            "msa_mask": torch.ones(N, L),
            "seq_mask": torch.ones(L),
            "pair_mask": torch.ones(L, L),
            "residx_atom14_to_atom37": protein["residx_atom14_to_atom37"],
            "residx_atom37_to_atom14": protein["residx_atom37_to_atom14"],
            "atom14_atom_exists": protein["atom14_atom_exists"],
            "atom37_atom_exists": protein["atom37_atom_exists"],
            "atom14_gt_exists": protein["atom14_gt_exists"],
            "atom14_gt_positions": protein["atom14_gt_positions"],
            "atom14_alt_gt_exists": protein["atom14_alt_gt_exists"],
            "atom14_alt_gt_positions": protein["atom14_alt_gt_positions"],
            "atom14_atom_is_ambiguous": protein["atom14_atom_is_ambiguous"],
            "template_aatype": torch.randint(0, 20, (T, L)),
            "template_all_atom_positions": torch.zeros(T, L, 37, 3),
            "template_all_atom_mask": torch.zeros(T, L, 37),
            "template_pseudo_beta": torch.zeros(T, L, 3),
            "template_pseudo_beta_mask": torch.zeros(T, L),
            "template_mask": torch.zeros(T),
            "template_torsion_angles_sin_cos": torch.zeros(T, L, 7, 2),
            "template_alt_torsion_angles_sin_cos": torch.zeros(T, L, 7, 2),
            "template_torsion_angles_mask": torch.zeros(T, L, 7),
        }

        features = {k: v.unsqueeze(-1) for k, v in features.items()}

        label = torch.randn(1)
        return features, label


def _collate(batch):
    features_list, labels = zip(*batch)
    batched = {}
    for key in features_list[0]:
        tensors = [f[key] for f in features_list]
        if tensors[0].dim() == 0:
            batched[key] = tensors[0]
        else:
            batched[key] = torch.stack(tensors)
    return batched, torch.stack(labels)


# ── Tests ─────────────────────────────────────────────────────────────

@skip_no_cuda
@skip_no_openfold
class TestOpenFoldReal:

    @pytest.fixture(autouse=True)
    def _setup(self):
        from openfold.config import model_config
        self.config = model_config("model_1_ptm")

    def test_adapter_loads_without_weights(self):
        """Build model from config, no weights — should work."""
        from molfun.backends.openfold import OpenFoldAdapter

        adapter = OpenFoldAdapter(config=self.config, device="cuda")
        info = adapter.param_summary()
        assert info["total"] > 90_000_000
        assert info["trainable"] == info["total"]

    @skip_no_weights
    def test_adapter_loads_with_weights(self):
        """Load real pre-trained weights."""
        from molfun.backends.openfold import OpenFoldAdapter

        adapter = OpenFoldAdapter(
            config=self.config,
            weights_path=str(WEIGHTS_PATH),
            device="cuda",
        )
        info = adapter.param_summary()
        assert info["total"] > 90_000_000

    @skip_no_weights
    def test_inference(self):
        """Run real inference with synthetic features."""
        from molfun.models.structure import MolfunStructureModel

        model = MolfunStructureModel(
            "openfold",
            config=self.config,
            weights=str(WEIGHTS_PATH),
            device="cuda",
        )

        ds = DummyOpenFoldDataset(n_samples=1, seq_len=16, n_msa=4)
        features, _ = ds[0]
        batch = {k: v.unsqueeze(0).cuda() if v.dim() > 0 else v for k, v in features.items()}

        out = model.predict(batch)
        assert out.single_repr is not None
        assert out.single_repr.shape[-1] == 384

    @skip_no_weights
    def test_freeze_and_peft(self):
        """Freeze trunk via LoRA strategy, verify param counts."""
        from molfun.models.structure import MolfunStructureModel
        from molfun.training import LoRAFinetune

        model = MolfunStructureModel(
            "openfold",
            config=self.config,
            weights=str(WEIGHTS_PATH),
            device="cuda",
            head="affinity",
            head_config={"single_dim": 384, "hidden_dim": 128},
        )

        strategy = LoRAFinetune(
            rank=4, alpha=8.0,
            target_modules=["linear_q", "linear_v"], use_hf=False,
        )
        strategy.setup(model)

        info = model.summary()
        assert info["adapter"]["frozen"] > 90_000_000
        assert info["peft"]["trainable_params"] > 0
        assert info["peft"]["trainable_pct"] < 1.0
        assert info["head"]["type"] == "AffinityHead"

    @skip_no_weights
    def test_fine_tune_one_step(self):
        """Run 1 training step with real OpenFold + LoRA + AffinityHead."""
        from molfun.models.structure import MolfunStructureModel
        from molfun.training import LoRAFinetune

        model = MolfunStructureModel(
            "openfold",
            config=self.config,
            weights=str(WEIGHTS_PATH),
            device="cuda",
            head="affinity",
            head_config={"single_dim": 384, "hidden_dim": 128},
        )

        strategy = LoRAFinetune(
            rank=4,
            target_modules=["linear_q", "linear_v"], use_hf=False,
            lr_head=1e-4, lr_lora=1e-4, amp=False,
        )

        ds = DummyOpenFoldDataset(n_samples=2, seq_len=16, n_msa=4)
        loader = DataLoader(ds, batch_size=1, collate_fn=_collate)

        history = model.fit(loader, strategy=strategy, epochs=1)
        assert len(history) == 1
        assert "train_loss" in history[0]
        assert history[0]["train_loss"] > 0

    @skip_no_weights
    def test_save_load_round_trip(self, tmp_path):
        """Save fine-tuned state, load into fresh model, verify consistency."""
        from molfun.models.structure import MolfunStructureModel
        from molfun.training import LoRAFinetune

        lora_kwargs = dict(rank=4, target_modules=["linear_q"], use_hf=False)
        head_config = {"single_dim": 384, "hidden_dim": 64}

        m1 = MolfunStructureModel(
            "openfold", config=self.config, weights=str(WEIGHTS_PATH),
            device="cuda", head="affinity", head_config=head_config,
        )
        s1 = LoRAFinetune(**lora_kwargs)
        s1.setup(m1)

        ckpt_dir = str(tmp_path / "real_ckpt")
        m1.save(ckpt_dir)

        m2 = MolfunStructureModel(
            "openfold", config=self.config, weights=str(WEIGHTS_PATH),
            device="cuda", head="affinity", head_config=head_config,
        )
        s2 = LoRAFinetune(**lora_kwargs)
        s2.setup(m2)
        m2.load(ckpt_dir)

        for (_, n1, l1), (_, n2, l2) in zip(
            m1._peft._builtin_layers, m2._peft._builtin_layers
        ):
            assert n1 == n2, f"Layer name mismatch: {n1} vs {n2}"
            assert torch.equal(l1.lora_A.data, l2.lora_A.data), f"lora_A mismatch at {n1}"
            assert torch.equal(l1.lora_B.data, l2.lora_B.data), f"lora_B mismatch at {n1}"

        for (k1, v1), (k2, v2) in zip(
            m1.head.state_dict().items(), m2.head.state_dict().items()
        ):
            assert torch.equal(v1, v2), f"Head weight mismatch at {k1}"
