"""
Tests for ESMBackend.
"""

import pytest
import torch

from molfun.models.backends.esm_backend import ESMBackend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestESMBackend:
    """Test suite for ESMBackend"""
    
    def test_init_esm2_8m(self, device, dtype):
        """Test ESMBackend initialization with ESM-2-8M"""
        backend = ESMBackend(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
        )
        assert backend.model_name == "ESM-2-8M"
        assert backend.dtype == dtype
        assert backend.device.type == device
    
    def test_init_esm2_150m(self, device, dtype):
        """Test ESMBackend initialization with ESM-2-150M"""
        backend = ESMBackend(
            model_name="ESM-2-150M",
            dtype=dtype,
            device=device,
        )
        assert backend.model_name == "ESM-2-150M"
    
    def test_lazy_loading(self, device, dtype):
        """Test that model loads lazily"""
        backend = ESMBackend(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
        )
        # Model should not be loaded until accessed
        assert backend._model is None
        assert backend._tokenizer is None
        
        # Accessing model property should trigger loading
        model = backend.model
        assert model is not None
        assert backend._model is not None
    
    def test_tokenizer_lazy_loading(self, device, dtype):
        """Test that tokenizer loads lazily"""
        backend = ESMBackend(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
        )
        assert backend._tokenizer is None
        
        tokenizer = backend.tokenizer
        assert tokenizer is not None
        assert backend._tokenizer is not None
    
    def test_eval(self, device, dtype):
        """Test eval mode"""
        backend = ESMBackend(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
        )
        result = backend.eval()
        assert result is backend
        assert backend._is_eval is True
        assert backend.model.training is False
    
    def test_infer_single(self, device, dtype, single_sequence):
        """Test inference with single sequence"""
        backend = ESMBackend(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
        )
        backend.eval()
        
        output = backend.infer(single_sequence)
        
        assert isinstance(output, torch.Tensor)
        assert output.dim() == 3
        assert output.shape[0] == 1
    
    def test_infer_batch(self, device, dtype, sample_sequences):
        """Test inference with batch"""
        backend = ESMBackend(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
        )
        backend.eval()
        
        output = backend.infer(sample_sequences)
        
        assert isinstance(output, torch.Tensor)
        assert output.dim() == 3
        assert output.shape[0] == len(sample_sequences)
    
    def test_infer_max_seq_length(self, device, dtype, single_sequence):
        """Test max_seq_length constraint"""
        backend = ESMBackend(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            max_seq_length=10,
        )
        backend.eval()
        
        output = backend.infer(single_sequence)
        assert output.shape[1] <= 12  # +2 for special tokens
    
    def test_direct_huggingface_id(self, device, dtype):
        """Test using direct HuggingFace model ID"""
        backend = ESMBackend(
            model_name="facebook/esm2_t6_8M_UR50D",
            dtype=dtype,
            device=device,
        )
        backend.eval()
        output = backend.infer("MKTAYIAKQR")
        assert output is not None
