"""GPU test: the released SimpleFold-100M weights load strictly through molfun."""

import pytest
import torch

pytest.importorskip("simplefold")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def test_load_100M_weights_strict():
    from molfun.models.structure import MolfunStructureModel

    model = MolfunStructureModel.from_pretrained(
        "simplefold_100M", device="cuda", head="generative"
    )

    trunk_params = sum(p.numel() for p in model.adapter.peft_target_module.parameters())
    assert 95e6 <= trunk_params <= 105e6
    assert model.adapter.sf.esm_model is not None
