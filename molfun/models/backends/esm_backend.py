"""
ESM model backend implementation.
"""

from __future__ import annotations
from typing import List, Optional, Union
import torch
from transformers import AutoTokenizer, EsmModel

from molfun.models.base_model import BaseBioModel


class ESMBackend(BaseBioModel):
    """ESM model backend implementation"""
    
    _SUPPORTED_MODELS = {
        "ESM-2-8M": "facebook/esm2_t6_8M_UR50D",
        "ESM-2-150M": "facebook/esm2_t30_150M_UR50D",
    }
    
    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.float16,
        max_seq_length: Optional[int] = None,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.device = torch.device(device)
        self._model: Optional[EsmModel] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._is_eval = False
        
    def _load(self):
        """Load model and tokenizer"""
        hf_model_id = self._SUPPORTED_MODELS.get(
            self.model_name,
            self.model_name  # Fallback to direct HuggingFace ID
        )
        
        self._tokenizer = AutoTokenizer.from_pretrained(hf_model_id, do_lower_case=False)
        self._model = EsmModel.from_pretrained(hf_model_id).to(self.device)
        self._model = self._model.to(self.dtype)
    
    @property
    def model(self) -> EsmModel:
        """Get the underlying model (lazy loading)"""
        if self._model is None:
            self._load()
        return self._model
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer (lazy loading)"""
        if self._tokenizer is None:
            self._load()
        return self._tokenizer
    
    def eval(self) -> ESMBackend:
        """Set model to evaluation mode"""
        self.model.eval()
        self._is_eval = True
        return self
    
    @torch.inference_mode()
    def infer(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        """
        Run inference on input sequences.
        
        Args:
            inputs: Single sequence string or list of sequence strings
            
        Returns:
            Tensor of shape [B, T, D] where B=batch, T=sequence_length, D=hidden_size
        """
        if not self._is_eval:
            self.eval()
        
        if isinstance(inputs, str):
            inputs = [inputs]
        
        batch = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        output = self.model(**batch)
        return output.last_hidden_state
