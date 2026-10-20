"""Sequence-only sampling through the SimpleFold backend (tiny CPU model)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("simplefold")

import molfun.models.structure as structure_mod  # noqa: E402
from molfun.backends.simplefold.config import SimpleFoldConfig  # noqa: E402
from molfun.backends.simplefold.imports import import_simplefold  # noqa: E402
from molfun.models.structure import MolfunStructureModel  # noqa: E402
from tests.toy_adapters import ToyGenerativeAdapter  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "simplefold"
TINY = SimpleFoldConfig(architecture=str(FIXTURES / "tiny_arch.yaml"), esm_model="esm2_8M")
SEQ = "MKTAYIAKQRQISFVKSHFS"  # 20 residues


@pytest.fixture(scope="module")
def adapter():
    from molfun.backends.simplefold.adapter import SimpleFoldAdapter

    return SimpleFoldAdapter(config=TINY, device="cpu")


def test_sample_uses_sequence_only(adapter, monkeypatch):
    sf = import_simplefold()
    original = sf.process_one_inference_structure
    seen = []

    def spy(structure_path, record_path, *args, **kwargs):
        seen.append(Path(structure_path))
        return original(structure_path, record_path, *args, **kwargs)

    monkeypatch.setattr(sf, "process_one_inference_structure", spy)

    adapter.sample(SEQ, num_samples=1, num_steps=2, seed=0)

    assert len(seen) == 1
    assert seen[0].name == "query.npz"
    assert seen[0].parents[1].name.startswith("molfun-simplefold-")


def test_sample_count_and_length(adapter):
    samples = adapter.sample(SEQ, num_samples=3, num_steps=5, seed=0)

    assert len(samples) == 3
    for s in samples:
        assert s.sequence == SEQ
        ca = s.ca_coords()
        assert ca.shape == (20, 3)
        assert not np.isnan(ca).any()


def test_sample_seed_reproducible(adapter):
    a = adapter.sample(SEQ, num_samples=1, num_steps=5, seed=7)[0].ca_coords()
    b = adapter.sample(SEQ, num_samples=1, num_steps=5, seed=7)[0].ca_coords()
    c = adapter.sample(SEQ, num_samples=1, num_steps=5, seed=8)[0].ca_coords()

    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_sampling_restores_training_mode(adapter):
    adapter.train()

    adapter.sample(SEQ, num_samples=1, num_steps=2, seed=0)

    assert adapter.training is True
    assert adapter.sf.esm_model.training is False


def test_model_sample_requires_generative_adapter(monkeypatch):
    registry = dict(structure_mod.ADAPTER_REGISTRY)
    registry["toy_generative"] = ToyGenerativeAdapter
    monkeypatch.setattr(structure_mod, "ADAPTER_REGISTRY", registry)
    model = MolfunStructureModel(name="toy_generative", device="cpu", head="generative")

    with pytest.raises(NotImplementedError, match="toy_generative"):
        model.sample(SEQ)
