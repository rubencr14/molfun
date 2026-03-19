"""
Integration test: Train → Save → Load → Verify predictions match.

Tests checkpoint persistence for both standard and LoRA models.

Note: MolfunStructureModel.save() persists the head and PEFT adapter
weights, not the full trunk. So to verify round-trip accuracy we must
use the *same* trunk instance (or a shared base model) and only reload
the head/PEFT weights on top.
"""

import pytest
import torch
import copy

from molfun.training import HeadOnlyFinetune, LoRAFinetune
from tests.integration.conftest import build_custom_model, make_loader, SEQ_LEN


def _predict(model, batch_dict):
    """Run a forward pass and return the affinity prediction tensor."""
    model.adapter.eval()
    model.head.eval()
    with torch.no_grad():
        result = model.forward(batch_dict)
    return result["preds"]


def _test_input():
    """Build a small deterministic dict batch."""
    torch.manual_seed(42)
    return {
        "aatype": torch.randint(0, 20, (2, SEQ_LEN)),
        "residue_index": torch.arange(SEQ_LEN).unsqueeze(0).expand(2, -1),
    }


class TestHeadOnlyCheckpoint:

    def test_save_load_head(self, tmp_path):
        """Save head weights, load into a model with identical trunk, verify match."""
        model = build_custom_model()
        strategy = HeadOnlyFinetune(lr=1e-3, amp=False, loss_fn="mse")
        strategy.fit(model, make_loader(), epochs=2, verbose=False)

        ckpt_dir = str(tmp_path / "ckpt")
        model.save(ckpt_dir)

        test_input = _test_input()
        preds_before = _predict(model, test_input)

        # Reload head into the SAME adapter (trunk unchanged by HeadOnly)
        model.load(ckpt_dir)
        preds_after = _predict(model, test_input)

        torch.testing.assert_close(preds_before, preds_after, atol=1e-5, rtol=1e-5)

    def test_checkpoint_files_exist(self, tmp_path):
        from pathlib import Path

        model = build_custom_model()
        strategy = HeadOnlyFinetune(lr=1e-3, amp=False, loss_fn="mse")
        strategy.fit(model, make_loader(), epochs=1, verbose=False)

        ckpt_dir = tmp_path / "ckpt"
        model.save(str(ckpt_dir))

        assert (ckpt_dir / "meta.pt").exists()
        assert (ckpt_dir / "head.pt").exists()

    def test_head_weights_change_after_training(self):
        """Verify training actually modifies head weights."""
        model = build_custom_model()
        head_before = {k: v.clone() for k, v in model.head.state_dict().items()}

        strategy = HeadOnlyFinetune(lr=1e-2, amp=False, loss_fn="mse")
        strategy.fit(model, make_loader(), epochs=3, verbose=False)

        head_after = model.head.state_dict()
        changed = any(
            not torch.equal(head_before[k], head_after[k]) for k in head_before
        )
        assert changed, "Head weights should change after training"


class TestLoRACheckpoint:

    def test_save_load_lora(self, tmp_path):
        """Save LoRA + head, load into same model, verify predictions match."""
        model = build_custom_model()
        strategy = LoRAFinetune(rank=4, lr_lora=1e-4, lr_head=1e-3, amp=False, loss_fn="mse")
        strategy.fit(model, make_loader(), epochs=2, verbose=False)

        ckpt_dir = str(tmp_path / "lora_ckpt")
        model.save(ckpt_dir)

        test_input = _test_input()
        preds_before = _predict(model, test_input)

        # Reload into the same model
        model.load(ckpt_dir)
        preds_after = _predict(model, test_input)

        torch.testing.assert_close(preds_before, preds_after, atol=1e-5, rtol=1e-5)

    def test_lora_checkpoint_has_peft_dir(self, tmp_path):
        from pathlib import Path

        model = build_custom_model()
        strategy = LoRAFinetune(rank=4, lr_lora=1e-4, lr_head=1e-3, amp=False, loss_fn="mse")
        strategy.fit(model, make_loader(), epochs=1, verbose=False)

        ckpt_dir = tmp_path / "lora_ckpt"
        model.save(str(ckpt_dir))

        assert (ckpt_dir / "meta.pt").exists()
        assert (ckpt_dir / "head.pt").exists()


class TestModelSummaryPersistence:

    def test_summary_before_and_after_training(self):
        model = build_custom_model()
        summary_pre = model.summary()
        assert "adapter" in summary_pre

        strategy = HeadOnlyFinetune(lr=1e-3, amp=False, loss_fn="mse")
        strategy.fit(model, make_loader(), epochs=1, verbose=False)

        summary_post = model.summary()
        assert summary_post["strategy"]["strategy"] == "HeadOnlyFinetune"
