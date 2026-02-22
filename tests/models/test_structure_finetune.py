"""
Structure prediction fine-tuning tests (no affinity labels needed).

Uses OpenFold's FAPE + aux losses directly against ground truth coordinates
embedded in the input batch. This is the "scientifically serious" use case:
specializing AlphaFold2 to a specific protein family without any labels.

Requires:
    - OpenFold installed
    - GPU with >= 16 GB VRAM
    - Weights at ~/.molfun/weights/finetuning_ptm_2.pt

Run:
    pytest tests/models/test_structure_finetune.py -v -s
"""

import pytest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

WEIGHTS_PATH = Path.home() / ".molfun" / "weights" / "finetuning_ptm_2.pt"


def _openfold_available() -> bool:
    try:
        from openfold.model.model import AlphaFold  # noqa: F401
        return True
    except ImportError:
        return False


skip_no_openfold = pytest.mark.skipif(
    not _openfold_available(), reason="OpenFold not installed"
)
skip_no_weights = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(), reason=f"Weights not found at {WEIGHTS_PATH}"
)
skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ── Synthetic dataset with GT ground truth ──────────────────────────────

class DummyStructureDataset(Dataset):
    """
    Produces OpenFold feature dicts with ground truth fields.

    The batch dict is passed directly to the model (no separate targets),
    which is the signature for structure fine-tuning.
    """

    def __init__(self, n_samples: int = 2, seq_len: int = 24, n_msa: int = 4):
        self.n = n_samples
        self.L = seq_len
        self.n_msa = n_msa

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        from openfold.data.data_transforms import (
            make_atom14_masks,
            make_atom14_positions,
            atom37_to_frames,
            get_backbone_frames,
            atom37_to_torsion_angles,
            get_chi_angles,
            make_pseudo_beta,
        )

        L, N = self.L, self.n_msa
        T = 1

        aatype = torch.randint(0, 20, (L,))

        # Build protein dict with all-atom placeholders
        protein = {
            "aatype": aatype,
            "all_atom_positions": torch.zeros(L, 37, 3),
            "all_atom_mask": torch.zeros(L, 37),
        }
        # atom14 masks and positions
        make_atom14_masks(protein)
        make_atom14_positions(protein)

        # Rigid group frames → backbone_rigid_tensor / _mask
        atom37_to_frames(protein)
        get_backbone_frames(protein)

        # Torsion angles → chi_angles_sin_cos / chi_mask
        atom37_to_torsion_angles("")(protein)
        get_chi_angles(protein)

        # pseudo-beta for distogram
        make_pseudo_beta("")(protein)

        # Input features (with trailing recycling dim)
        feat = {
            "aatype":                       aatype,
            "residue_index":               torch.arange(L),
            "msa_feat":                    torch.randn(N, L, 49),
            "target_feat":                 torch.randn(L, 22),
            "extra_msa":                   torch.randint(0, 23, (N * 4, L)),
            "extra_has_deletion":          torch.zeros(N * 4, L),
            "extra_deletion_value":        torch.zeros(N * 4, L),
            "extra_msa_mask":              torch.ones(N * 4, L),
            "msa_mask":                    torch.ones(N, L),
            "seq_mask":                    torch.ones(L),
            "pair_mask":                   torch.ones(L, L),
            # atom14 / atom37 lookup tables + masks
            "residx_atom14_to_atom37":     protein["residx_atom14_to_atom37"],
            "residx_atom37_to_atom14":     protein["residx_atom37_to_atom14"],
            "atom14_atom_exists":          protein["atom14_atom_exists"],
            "atom37_atom_exists":          protein["atom37_atom_exists"],
            "all_atom_positions":          torch.zeros(L, 37, 3),
            "all_atom_mask":               torch.zeros(L, 37),
            # Ground truth fields (for AlphaFoldLoss)
            "atom14_gt_exists":            protein["atom14_gt_exists"],
            "atom14_gt_positions":         protein["atom14_gt_positions"],
            "atom14_alt_gt_exists":        protein["atom14_alt_gt_exists"],
            "atom14_alt_gt_positions":     protein["atom14_alt_gt_positions"],
            "atom14_atom_is_ambiguous":    protein["atom14_atom_is_ambiguous"],
            "backbone_rigid_tensor":       protein["backbone_rigid_tensor"],
            "backbone_rigid_mask":         protein["backbone_rigid_mask"],
            # Rigid group frames (for sidechain FAPE)
            "rigidgroups_gt_frames":       protein["rigidgroups_gt_frames"],
            "rigidgroups_alt_gt_frames":   protein["rigidgroups_alt_gt_frames"],
            "rigidgroups_gt_exists":       protein["rigidgroups_gt_exists"],
            # Torsion / chi angles
            "chi_angles_sin_cos":          protein["chi_angles_sin_cos"],
            "chi_mask":                    protein["chi_mask"],
            "pseudo_beta":                 protein["pseudo_beta"],
            "pseudo_beta_mask":            protein["pseudo_beta_mask"],
            # Template placeholders
            "template_aatype":             torch.randint(0, 20, (T, L)),
            "template_all_atom_positions": torch.zeros(T, L, 37, 3),
            "template_all_atom_mask":      torch.zeros(T, L, 37),
            "template_pseudo_beta":        torch.zeros(T, L, 3),
            "template_pseudo_beta_mask":   torch.zeros(T, L),
            "template_mask":               torch.zeros(T),
            "template_torsion_angles_sin_cos":     torch.zeros(T, L, 7, 2),
            "template_alt_torsion_angles_sin_cos": torch.zeros(T, L, 7, 2),
            "template_torsion_angles_mask":        torch.zeros(T, L, 7),
        }

        # All tensors get a trailing recycling dimension R=1
        # (OpenFold peels it off in its recycling loop)
        out = {k: v.unsqueeze(-1) for k, v in feat.items()}

        # seq_length: needed by AlphaFoldLoss for loss scaling
        out["seq_length"] = torch.tensor([L], dtype=torch.float32)

        # Return bare dict (no targets): this is structure fine-tuning mode
        return out


def _collate_structure(batch):
    """Stack list of feature dicts into a batched dict."""
    out = {}
    for key in batch[0]:
        tensors = [b[key] for b in batch]
        if tensors[0].dim() == 0:
            out[key] = tensors[0]
        else:
            out[key] = torch.stack(tensors)
    return out


# ── Tests ────────────────────────────────────────────────────────────────

@skip_no_cuda
@skip_no_openfold
@skip_no_weights
class TestStructureFinetune:

    @pytest.fixture(autouse=True)
    def _setup(self):
        from openfold.config import model_config
        self.config = model_config("model_1_ptm")

    def _make_loader(self, n=2, batch_size=1):
        ds = DummyStructureDataset(n_samples=n, seq_len=24)
        return DataLoader(
            ds, batch_size=batch_size, collate_fn=_collate_structure
        )

    def test_structure_head_init(self):
        """StructureLossHead initialises from model config."""
        from molfun.heads.structure import StructureLossHead
        head = StructureLossHead(self.config.loss)
        assert hasattr(head, "_loss_fn")

    def test_structure_head_fape_only(self):
        from molfun.heads.structure import StructureLossHead
        head = StructureLossHead.fape_only(self.config)
        assert head._loss_config.masked_msa.weight == 0.0
        assert head._loss_config.fape.weight == 1.0

    def test_structure_head_with_weights(self):
        from molfun.heads.structure import StructureLossHead
        head = StructureLossHead.with_weights(
            self.config, masked_msa=0.0, distogram=0.0
        )
        assert head._loss_config.masked_msa.weight == 0.0

    def test_model_structure_head_registered(self):
        """MolfunStructureModel accepts head='structure'."""
        from molfun.models.structure import MolfunStructureModel
        from molfun.heads.structure import StructureLossHead
        model = MolfunStructureModel(
            "openfold",
            config=self.config,
            weights=str(WEIGHTS_PATH),
            device="cuda",
            head="structure",
            head_config={"loss_config": self.config.loss},
        )
        assert isinstance(model.head, StructureLossHead)

    def test_partial_finetune_structure_one_step(self):
        """PartialFinetune trains on structure loss for one step."""
        from molfun.models.structure import MolfunStructureModel
        from molfun.training.partial import PartialFinetune

        model = MolfunStructureModel(
            "openfold",
            config=self.config,
            weights=str(WEIGHTS_PATH),
            device="cuda",
            head="structure",
            head_config={"loss_config": self.config.loss},
        )

        strategy = PartialFinetune(
            unfreeze_last_n=2,
            unfreeze_structure_module=True,
            lr_trunk=1e-5,
            amp=False,  # avoid fp16 instability with synthetic data
            early_stopping_patience=0,
        )

        loader = self._make_loader(n=2, batch_size=1)
        history = model.fit(loader, strategy=strategy, epochs=1)

        assert len(history) == 1
        assert "train_loss" in history[0]
        assert torch.isfinite(torch.tensor(history[0]["train_loss"]))

    def test_lora_finetune_structure(self):
        """LoRAFinetune trains on structure loss for one step."""
        from molfun.models.structure import MolfunStructureModel
        from molfun.training.lora import LoRAFinetune

        model = MolfunStructureModel(
            "openfold",
            config=self.config,
            weights=str(WEIGHTS_PATH),
            device="cuda",
            head="structure",
            head_config={"loss_config": self.config.loss},
        )

        strategy = LoRAFinetune(
            rank=4,
            lr_lora=1e-4,
            amp=False,
        )

        loader = self._make_loader(n=2, batch_size=1)
        history = model.fit(loader, strategy=strategy, epochs=1)

        assert len(history) == 1
        assert torch.isfinite(torch.tensor(history[0]["train_loss"]))
