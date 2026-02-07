"""
Base classes for biological language models.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import torch


class BaseBioModel(ABC):
    """Abstract base class for biological language models"""
    
    @abstractmethod
    def eval(self) -> BaseBioModel:
        """Set model to evaluation mode"""
        pass
    
    @abstractmethod
    def infer(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        """
        Run inference on input sequences.
        
        Args:
            inputs: Single sequence string or list of sequence strings
            
        Returns:
            Tensor embeddings
        """
        pass


class ModelOptimizer(ABC):
    """Abstract interface for applying optimizations to models"""
    
    @abstractmethod
    def apply(self, model: torch.nn.Module) -> List:
        """
        Apply optimizations to model.
        
        Returns:
            List of restore functions to undo patches
        """
        pass
    
    @abstractmethod
    def restore(self, model: torch.nn.Module, patches: List):
        """Restore original model state"""
        pass
