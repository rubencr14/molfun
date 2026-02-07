"""
Optimized BioLanguageModel for protein language modeling with Triton kernels.
"""

from __future__ import annotations
from typing import List, Optional, Union, Dict, Type
from dataclasses import dataclass

import torch

from molfun.models.base_model import BaseBioModel, ModelOptimizer
from molfun.models.backends.esm_backend import ESMBackend
from molfun.models.optimizers.esm_optimizer import ESMOptimizer


@dataclass
class ModelConfig:
    """Configuration for BioLanguageModel"""
    model_name: str
    dtype: torch.dtype = torch.float16
    max_seq_length: Optional[int] = None
    device: str = "cuda"
    use_optimizations: bool = True


class BioLanguageModel(BaseBioModel):
    """
    Optimized biological language model with Triton kernel optimizations.
    
    Supports multiple model families (ESM, AlphaFold, DiffDock, etc.) with
    optional fused operations for improved performance.
    """
    
    _BACKEND_REGISTRY: Dict[str, Type[BaseBioModel]] = {
        "ESM-2-8M": ESMBackend,
        "ESM-2-150M": ESMBackend,
    }
    
    _OPTIMIZER_REGISTRY: Dict[str, Type[ModelOptimizer]] = {
        "ESM-2-8M": ESMOptimizer,
        "ESM-2-150M": ESMOptimizer,
    }
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._backend: Optional[BaseBioModel] = None
        self._optimizer: Optional[ModelOptimizer] = None
        self._patches_restore = []
        
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        dtype: torch.dtype = torch.float16,
        max_seq_length: Optional[int] = None,
        device: str = "cuda",
        use_optimizations: bool = True,
    ) -> BioLanguageModel:
        """
        Create a BioLanguageModel from a pretrained model.
        
        Args:
            model_name: Model identifier (e.g., "ESM-2-8M", "ESM-2-150M")
            dtype: Data type for inference (torch.float16 or torch.bfloat16)
            max_seq_length: Maximum sequence length (None for model default)
            device: Device to run on ("cuda" or "cpu")
            use_optimizations: Whether to apply Triton kernel optimizations
            
        Returns:
            BioLanguageModel instance
        """
        config = ModelConfig(
            model_name=model_name,
            dtype=dtype,
            max_seq_length=max_seq_length,
            device=device,
            use_optimizations=use_optimizations,
        )
        instance = cls(config)
        instance._initialize_backend()
        return instance
    
    def _initialize_backend(self):
        """Initialize the appropriate backend for the model"""
        # Determine backend type from model name
        backend_class = None
        for key, backend in self._BACKEND_REGISTRY.items():
            if self.config.model_name.startswith(key.split("-")[0]):  # Match "ESM", "AlphaFold", etc.
                backend_class = backend
                break
        
        if backend_class is None:
            # Default to ESM for known ESM models
            if "esm" in self.config.model_name.lower() or "ESM" in self.config.model_name:
                backend_class = ESMBackend
            else:
                raise ValueError(f"Unknown model type: {self.config.model_name}")
        
        # Create backend instance
        self._backend = backend_class(
            model_name=self.config.model_name,
            dtype=self.config.dtype,
            max_seq_length=self.config.max_seq_length,
            device=self.config.device,
        )
        
        # Apply optimizations if requested
        if self.config.use_optimizations:
            optimizer_class = self._OPTIMIZER_REGISTRY.get(
                self.config.model_name,
                None
            )
            
            if optimizer_class is None:
                # Try to match by prefix
                for key, opt_class in self._OPTIMIZER_REGISTRY.items():
                    if self.config.model_name.startswith(key.split("-")[0]):
                        optimizer_class = opt_class
                        break
            
            if optimizer_class:
                self._optimizer = optimizer_class()
                # Get the underlying model to apply optimizations
                if hasattr(self._backend, 'model'):
                    model = self._backend.model
                    self._patches_restore = self._optimizer.apply(model)
    
    def eval(self) -> BioLanguageModel:
        """Set model to evaluation mode"""
        if self._backend is not None:
            self._backend.eval()
        return self
    
    def infer(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        """
        Run inference on input sequences.
        
        Args:
            inputs: Single sequence string or list of sequence strings
            
        Returns:
            Tensor embeddings
        """
        if self._backend is None:
            raise RuntimeError("Model not initialized")
        return self._backend.infer(inputs)
    
    def __del__(self):
        """Cleanup patches on deletion"""
        if self._optimizer and self._patches_restore and hasattr(self._backend, 'model'):
            self._optimizer.restore(self._backend.model, self._patches_restore)
