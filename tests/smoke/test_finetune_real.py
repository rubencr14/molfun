"""
Real end-to-end fine-tuning smoke tests.

These tests load actual pretrained models, fetch real PDB structures,
and run short training loops on GPU. They validate the full pipeline
works — from data to training to checkpoint save/load.

Run with:
    pytest tests/smoke/test_finetune_real.py -v -s

Skip if no GPU:
    pytest tests/smoke/test_finetune_real.py -v -s -m gpu
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.disable(logging.WARNING)

import torch
from torch.utils.data import DataLoader

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def kinase_paths():
    """Fetch 5 small kinase structures (cached across tests in this module)."""
    from molfun.data.collections import fetch_collection

    paths = fetch_collection(
        "kinases_human",
        max_structures=5,
        resolution_max=2.5,
        cache_dir="./data/kinases",
    )
    assert len(paths) > 0, "Failed to fetch any kinase structures"
    return paths


@pytest.fixture(scope="module")
def openfold_model():
    """Load OpenFold model once for the module."""
    from molfun.models import MolfunStructureModel

    model = MolfunStructureModel.from_pretrained(
        "openfold", device="cuda", head="structure",
    )
    return model


# ─── Structure fine-tuning (LoRA) ────────────────────────────────────


@gpu
class TestOpenFoldLoRAFinetune:
    """LoRA fine-tuning on structure prediction with real OpenFold."""

    def test_full_pipeline(self, kinase_paths, openfold_model, tmp_path):
        from molfun.backends.openfold import OpenFoldFeaturizer
        from molfun.data.datasets import StructureDataset
        from molfun.data.splits import DataSplitter
        from molfun.training import LoRAFinetune
        from molfun.tracking import ExperimentRegistry

        model = openfold_model

        featurizer = OpenFoldFeaturizer(
            config=model.adapter.model.config, max_seq_len=128,
        )
        dataset = StructureDataset(pdb_paths=kinase_paths, featurizer=featurizer)
        train_ds, val_ds, _ = DataSplitter.random(
            dataset, val_frac=0.2, test_frac=0.0, seed=0,
        )

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

        strategy = LoRAFinetune(
            rank=4, alpha=8,
            lr_lora=1e-4, lr_head=1e-3,
            accumulation_steps=2, warmup_steps=2,
            amp=False, ema_decay=0.0,
        )

        registry = ExperimentRegistry(local_dir=str(tmp_path / "registry"))
        registry.start_run(
            name="test-lora-structure",
            tags=["test", "lora", "structure"],
            config={"rank": 4, "epochs": 2, "seq_len": 128},
        )
        registry.log_pdb_refs(kinase_paths)

        history = model.fit(
            train_loader, val_loader,
            strategy=strategy,
            epochs=2,
            gradient_checkpointing=True,
            tracker=registry,
        )

        assert len(history) == 2
        assert all("train_loss" in h for h in history)
        assert all("val_loss" in h for h in history)
        assert all(h["train_loss"] > 0 for h in history)
        assert history[-1]["train_loss"] <= history[0]["train_loss"] + 5.0

        ckpt_path = str(tmp_path / "ckpt_lora")
        model.save(ckpt_path)
        assert (tmp_path / "ckpt_lora" / "meta.pt").exists()
        assert (tmp_path / "ckpt_lora" / "head.pt").exists()

        registry.end_run()
        manifest_path = registry.run_dir / "manifest.json"
        assert manifest_path.exists()

    def test_summary_after_lora(self, openfold_model):
        info = openfold_model.summary()
        assert info["adapter"]["total"] > 0
        assert info["adapter"]["trainable"] > 0


@gpu
class TestOpenFoldHeadOnlyFinetune:
    """Head-only fine-tuning (trunk frozen) with real OpenFold."""

    def test_head_only_trains(self, kinase_paths, tmp_path):
        from molfun.models import MolfunStructureModel
        from molfun.backends.openfold import OpenFoldFeaturizer
        from molfun.data.datasets import StructureDataset
        from molfun.data.splits import DataSplitter
        from molfun.training import HeadOnlyFinetune

        model = MolfunStructureModel.from_pretrained(
            "openfold", device="cuda", head="structure",
        )

        featurizer = OpenFoldFeaturizer(
            config=model.adapter.model.config, max_seq_len=128,
        )
        dataset = StructureDataset(pdb_paths=kinase_paths, featurizer=featurizer)
        train_ds, val_ds, _ = DataSplitter.random(
            dataset, val_frac=0.2, test_frac=0.0, seed=0,
        )

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

        strategy = HeadOnlyFinetune(lr=1e-3, amp=False)

        history = model.fit(
            train_loader, val_loader,
            strategy=strategy,
            epochs=2,
            gradient_checkpointing=True,
        )

        assert len(history) == 2
        assert all("train_loss" in h for h in history)


# ─── Affinity head fine-tuning (from CSV) ────────────────────────────

AFFINITY_CSV = Path(__file__).parent / "fixtures" / "kinase_affinity.csv"
PDB_DIR = Path("./data/kinases")


@gpu
class TestOpenFoldAffinityFinetune:
    """
    Fine-tuning with AffinityHead from a real CSV of pKd values.

    Uses kinase_affinity.csv (5 PDB IDs + pKd labels) paired with
    the same CIF files fetched for structure tests.
    """

    def test_affinity_from_csv(self, kinase_paths, tmp_path):
        """Load labels from CSV, train AffinityHead with LoRA on real structures."""
        import csv
        from molfun.models import MolfunStructureModel
        from molfun.backends.openfold import OpenFoldFeaturizer
        from molfun.data.datasets import StructureDataset
        from molfun.data.splits import DataSplitter
        from molfun.training import LoRAFinetune

        labels = {}
        with open(AFFINITY_CSV) as f:
            for row in csv.DictReader(f):
                labels[row["pdb_id"].strip().lower()] = float(row["affinity"])

        assert len(labels) >= 3, f"CSV has only {len(labels)} entries"

        model = MolfunStructureModel.from_pretrained(
            "openfold", device="cuda",
            head="affinity", head_config={"single_dim": 384},
        )

        featurizer = OpenFoldFeaturizer(
            config=model.adapter.model.config, max_seq_len=128,
        )

        dataset = StructureDataset(
            pdb_paths=kinase_paths,
            featurizer=featurizer,
            labels=labels,
        )
        train_ds, val_ds, _ = DataSplitter.random(
            dataset, val_frac=0.2, test_frac=0.0, seed=0,
        )

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

        strategy = LoRAFinetune(
            rank=4, alpha=8,
            lr_lora=1e-4, lr_head=1e-3,
            accumulation_steps=1, warmup_steps=2,
            amp=False, ema_decay=0.0,
            loss_fn="mse",
        )

        history = model.fit(
            train_loader, val_loader,
            strategy=strategy,
            epochs=2,
            gradient_checkpointing=True,
        )

        assert len(history) == 2
        assert all("train_loss" in h for h in history)
        assert all(h["train_loss"] >= 0 for h in history)

        ckpt_path = str(tmp_path / "ckpt_affinity")
        model.save(ckpt_path)
        assert (tmp_path / "ckpt_affinity" / "head.pt").exists()
        assert (tmp_path / "ckpt_affinity" / "meta.pt").exists()

    @pytest.mark.parametrize("loss_fn", ["mse", "huber", "mae"])
    def test_affinity_loss_variants(self, kinase_paths, loss_fn):
        """Each loss function completes 1 epoch without error."""
        import csv
        from molfun.models import MolfunStructureModel
        from molfun.backends.openfold import OpenFoldFeaturizer
        from molfun.data.datasets import StructureDataset
        from molfun.training import LoRAFinetune

        labels = {}
        with open(AFFINITY_CSV) as f:
            for row in csv.DictReader(f):
                labels[row["pdb_id"].strip().lower()] = float(row["affinity"])

        model = MolfunStructureModel.from_pretrained(
            "openfold", device="cuda",
            head="affinity", head_config={"single_dim": 384},
        )
        featurizer = OpenFoldFeaturizer(
            config=model.adapter.model.config, max_seq_len=128,
        )
        dataset = StructureDataset(
            pdb_paths=kinase_paths, featurizer=featurizer, labels=labels,
        )
        loader = DataLoader(dataset, batch_size=1, num_workers=0)

        strategy = LoRAFinetune(
            rank=4, alpha=8, lr_lora=1e-4, lr_head=1e-3,
            amp=False, loss_fn=loss_fn,
        )
        history = model.fit(loader, epochs=1, strategy=strategy, gradient_checkpointing=True)

        assert len(history) == 1
        assert history[0]["train_loss"] >= 0


# ─── Checkpoint save/load roundtrip ──────────────────────────────────


@gpu
class TestCheckpointRoundtrip:
    """Verify model weights survive a save → load cycle."""

    def test_save_load_preserves_head(self, kinase_paths, tmp_path):
        import csv as _csv
        from molfun.models import MolfunStructureModel
        from molfun.backends.openfold import OpenFoldFeaturizer
        from molfun.data.datasets import StructureDataset
        from molfun.training import LoRAFinetune

        labels = {}
        with open(AFFINITY_CSV) as f:
            for row in _csv.DictReader(f):
                labels[row["pdb_id"].strip().lower()] = float(row["affinity"])

        model = MolfunStructureModel.from_pretrained(
            "openfold", device="cuda",
            head="affinity", head_config={"single_dim": 384},
        )

        featurizer = OpenFoldFeaturizer(
            config=model.adapter.model.config, max_seq_len=128,
        )
        dataset = StructureDataset(
            pdb_paths=kinase_paths, featurizer=featurizer, labels=labels,
        )
        loader = DataLoader(dataset, batch_size=1, num_workers=0)

        strategy = LoRAFinetune(
            rank=4, alpha=8, lr_lora=1e-4, lr_head=1e-3, amp=False,
        )
        model.fit(loader, epochs=1, strategy=strategy, gradient_checkpointing=True)

        ckpt = str(tmp_path / "roundtrip")
        model.save(ckpt)

        model2 = MolfunStructureModel.from_pretrained(
            "openfold", device="cuda",
            head="affinity", head_config={"single_dim": 384},
        )
        strategy2 = LoRAFinetune(rank=4, alpha=8, amp=False)
        strategy2.setup(model2)
        model2.load(ckpt)

        for (n1, p1), (n2, p2) in zip(
            model.head.state_dict().items(),
            model2.head.state_dict().items(),
        ):
            assert n1 == n2
            assert torch.allclose(p1.cpu(), p2.cpu()), f"Mismatch in head param: {n1}"


# ─── Registry integration ────────────────────────────────────────────


@gpu
class TestRegistryWithTraining:
    """ExperimentRegistry tracks a real training run."""

    def test_registry_captures_metrics(self, kinase_paths, openfold_model, tmp_path):
        import json
        from molfun.backends.openfold import OpenFoldFeaturizer
        from molfun.data.datasets import StructureDataset
        from molfun.data.splits import DataSplitter
        from molfun.training import LoRAFinetune
        from molfun.tracking import ExperimentRegistry

        model = openfold_model

        featurizer = OpenFoldFeaturizer(
            config=model.adapter.model.config, max_seq_len=128,
        )
        dataset = StructureDataset(pdb_paths=kinase_paths, featurizer=featurizer)
        train_ds, val_ds, _ = DataSplitter.random(
            dataset, val_frac=0.2, test_frac=0.0, seed=0,
        )

        train_loader = DataLoader(train_ds, batch_size=1, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

        strategy = LoRAFinetune(
            rank=4, alpha=8, lr_lora=1e-4, lr_head=1e-3,
            accumulation_steps=2, warmup_steps=2, amp=False,
        )

        registry = ExperimentRegistry(local_dir=str(tmp_path / "reg"))
        registry.start_run(name="registry-test", config={"rank": 4})
        registry.log_pdb_refs(kinase_paths)

        model.fit(
            train_loader, val_loader,
            strategy=strategy, epochs=2,
            gradient_checkpointing=True, tracker=registry,
        )
        registry.end_run()

        manifest = json.loads((registry.run_dir / "manifest.json").read_text())
        metrics = json.loads((registry.run_dir / "metrics.json").read_text())
        pdb_refs = json.loads((registry.run_dir / "pdb_refs.json").read_text())

        assert manifest["status"] == "completed"
        assert manifest["elapsed_seconds"] > 0
        assert len(metrics) >= 2
        assert len(pdb_refs) == len(kinase_paths)
        assert all("train_loss" in m for m in metrics)
