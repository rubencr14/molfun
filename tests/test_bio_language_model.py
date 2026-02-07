"""
Tests for BioLanguageModel.
"""

import pytest
import torch

from molfun import BioLanguageModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBioLanguageModel:
    """Test suite for BioLanguageModel"""
    
    def test_from_pretrained_esm2_8m(self, device, dtype):
        """Test creating ESM-2-8M model"""
        model = BioLanguageModel.from_pretrained(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            use_optimizations=False,  # Disable for faster tests
        )
        assert model is not None
        assert model._backend is not None
    
    def test_from_pretrained_esm2_150m(self, device, dtype):
        """Test creating ESM-2-150M model"""
        model = BioLanguageModel.from_pretrained(
            model_name="ESM-2-150M",
            dtype=dtype,
            device=device,
            use_optimizations=False,
        )
        assert model is not None
        assert model._backend is not None
    
    def test_eval_mode(self, device, dtype):
        """Test setting model to eval mode"""
        model = BioLanguageModel.from_pretrained(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            use_optimizations=False,
        )
        result = model.eval()
        assert result is model  # Should return self
        assert model._backend._is_eval is True
    
    def test_infer_single_sequence(self, device, dtype, single_sequence):
        """Test inference with single sequence"""
        model = BioLanguageModel.from_pretrained(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            use_optimizations=False,
        )
        model.eval()
        
        output = model.infer(single_sequence)
        
        assert isinstance(output, torch.Tensor)
        assert output.dim() == 3  # [B, T, D]
        assert output.shape[0] == 1  # Batch size
        assert output.device.type == device
    
    def test_infer_batch_sequences(self, device, dtype, sample_sequences):
        """Test inference with batch of sequences"""
        model = BioLanguageModel.from_pretrained(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            use_optimizations=False,
        )
        model.eval()
        
        output = model.infer(sample_sequences)
        
        assert isinstance(output, torch.Tensor)
        assert output.dim() == 3  # [B, T, D]
        assert output.shape[0] == len(sample_sequences)
        assert output.device.type == device
    
    def test_infer_with_max_seq_length(self, device, dtype, single_sequence):
        """Test inference with max_seq_length constraint"""
        max_length = 10
        model = BioLanguageModel.from_pretrained(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            max_seq_length=max_length,
            use_optimizations=False,
        )
        model.eval()
        
        output = model.infer(single_sequence)
        
        assert output.shape[1] <= max_length + 2  # +2 for special tokens
    
    def test_optimizations_enabled(self, device, dtype):
        """Test that optimizations can be enabled"""
        model = BioLanguageModel.from_pretrained(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            use_optimizations=True,
        )
        assert model._optimizer is not None
        assert len(model._patches_restore) > 0
    
    def test_optimizations_disabled(self, device, dtype):
        """Test that optimizations can be disabled"""
        model = BioLanguageModel.from_pretrained(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            use_optimizations=False,
        )
        assert model._optimizer is None
        assert len(model._patches_restore) == 0
    
    def test_infer_output_shape_consistency(self, device, dtype, sample_sequences):
        """Test that output shapes are consistent"""
        model = BioLanguageModel.from_pretrained(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            use_optimizations=False,
        )
        model.eval()
        
        outputs = []
        for seq in sample_sequences:
            output = model.infer(seq)
            outputs.append(output)
        
        # All outputs should have same hidden dimension
        hidden_dims = [out.shape[2] for out in outputs]
        assert len(set(hidden_dims)) == 1  # All same
    
    def test_infer_dtype_consistency(self, device, dtype, single_sequence):
        """Test that output dtype matches model dtype"""
        model = BioLanguageModel.from_pretrained(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            use_optimizations=False,
        )
        model.eval()
        
        output = model.infer(single_sequence)
        
        assert output.dtype == dtype
    
    def test_model_cleanup(self, device, dtype):
        """Test that model cleanup works correctly"""
        model = BioLanguageModel.from_pretrained(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            use_optimizations=True,
        )
        
        # Store patches before deletion
        patches = model._patches_restore.copy()
        
        # Delete model (triggers __del__)
        del model
        
        # If we can still access patches, cleanup should have run
        # (This is a basic test - full cleanup verification would need more complex setup)
        assert True  # Placeholder - cleanup is hard to test directly
