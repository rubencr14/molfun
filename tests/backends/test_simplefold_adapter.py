"""Unit tests for the SimpleFold backend on CPU, with a tiny architecture
(tests/fixtures/simplefold/tiny_arch.yaml) and the processed 1CRN record."""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

pytest.importorskip("simplefold")

from molfun.backends.simplefold.config import SimpleFoldConfig  # noqa: E402
from molfun.backends.simplefold.imports import import_simplefold  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "simplefold"
RECORD = FIXTURES / "tiny_record"
TINY = SimpleFoldConfig(
    architecture=str(FIXTURES / "tiny_arch.yaml"), esm_model="esm2_8M", multiplicity=2
)


@pytest.fixture(scope="module")
def adapter():
    from molfun.backends.simplefold.adapter import SimpleFoldAdapter

    return SimpleFoldAdapter(config=TINY, device="cpu")


@pytest.fixture(scope="module")
def batch():
    from molfun.backends.simplefold.records import SimpleFoldRecordDataset, collate_records

    ds = SimpleFoldRecordDataset(
        tokenized_dir=RECORD / "tokens", target_dir=RECORD / "targets", groups={"crn": ["1crn"]}
    )
    return collate_records([ds[0]])


def test_imports_coexist_with_hf_datasets():
    hf = pytest.importorskip("datasets")
    import_simplefold.cache_clear()

    ns = import_simplefold()

    assert sys.modules["datasets"] is hf
    assert hasattr(ns, "SimpleFold")


def test_only_architecture_is_trainable(adapter):
    adapter.unfreeze_trunk()

    trainable = {id(p) for p in adapter.parameters() if p.requires_grad}
    architecture = {id(p) for p in adapter.peft_target_module.parameters()}

    assert trainable == architecture
    assert not any(p.requires_grad for p in adapter.sf.esm_model.parameters())


def test_freeze_trunk_freezes_architecture(adapter):
    adapter.freeze_trunk()

    assert not any(p.requires_grad for p in adapter.parameters())
    adapter.unfreeze_trunk()


def test_trunk_blocks_are_the_residue_trunk(adapter):
    assert (
        len(adapter.get_trunk_blocks()) == 2
    )  # trunk depth in tiny_arch.yaml (atom trunks have 1)


def _matched(targets, modules):
    """Modules HF PEFT would target: name equal to a target or ending in '.<target>'."""
    return {
        name: module
        for name, module in modules
        if any(name == t or name.endswith("." + t) for t in targets)
    }


def test_default_peft_targets_match_only_linear_layers(adapter):
    matched = _matched(adapter.default_peft_targets, adapter.peft_target_module.named_modules())

    assert matched
    assert all(isinstance(m, nn.Linear) for m in matched.values()), [
        n for n, m in matched.items() if not isinstance(m, nn.Linear)
    ]
    leaf_names = {name.rsplit(".", 1)[-1] for name in matched}
    assert leaf_names == {"qkv", "proj", "w1", "w2"}
    assert any(name.startswith("trunk.blocks.") for name in matched)


def test_esm_stays_in_eval_mode_when_training(adapter):
    adapter.train()

    assert adapter.sf.esm_model.training is False


def test_forward_returns_three_losses(adapter, batch):
    torch.manual_seed(0)
    adapter.train()

    losses = adapter(batch).extra["losses"]

    assert set(losses) == {"total", "flow_mse", "smooth_lddt"}
    assert all(math.isfinite(v.item()) for v in losses.values())
    assert losses["total"].item() == pytest.approx(
        losses["flow_mse"].item() + losses["smooth_lddt"].item(), rel=1e-4
    )
    assert losses["total"].requires_grad


def test_dataset_samples_by_group():
    from molfun.backends.simplefold.records import SimpleFoldRecordDataset

    ds = SimpleFoldRecordDataset(
        tokenized_dir=RECORD / "tokens",
        target_dir=RECORD / "targets",
        groups={"big": ["1crn"] * 50, "small": ["1crn"]},
    )

    assert len(ds) == 2
    item = ds[1]
    assert {"coords", "aa_seq", "record"} <= set(item)


def test_hub_config_for_simplefold_100m():
    from molfun.hub.registry import PRETRAINED_REGISTRY, get_config

    spec = PRETRAINED_REGISTRY["simplefold_100M"]

    assert spec.backend == "simplefold"
    assert get_config("simplefold_100M") == SimpleFoldConfig(architecture="foldingdit_100M")
