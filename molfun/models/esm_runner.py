from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
from transformers import AutoTokenizer, EsmModel

@dataclass
class ESMConfig:
    model_id: str = "facebook/esm2_t6_8M_UR50D"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

class ESMRunnerHF:
    def __init__(self, cfg: ESMConfig = ESMConfig()):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, do_lower_case=False)
        self.model = EsmModel.from_pretrained(cfg.model_id).to(self.device).eval()

        # inference dtype (keep fp32 if you want max numeric stability)
        self.model = self.model.to(cfg.dtype)

    @torch.inference_mode()
    def embed_per_residue(self, sequences: List[str]) -> torch.Tensor:
        """
        sequences: list of amino-acid strings, e.g. ["MKT...", "ACDE..."]
        returns: last_hidden_state [B, T, D]
        """
        batch = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        out = self.model(**batch)
        return out.last_hidden_state  # [B, T, D]
