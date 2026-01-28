"""
Tests for ESMOptimizer.
"""

import pytest
import torch
from transformers import EsmModel

from molfun.models.optimizers.esm_optimizer import ESMOptimizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestESMOptimizer:
    """Test suite for ESMOptimizer"""
    
    @pytest.fixture
    def model(self, device, dtype):
        """Create a test ESM model"""
        model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        model = model.to(device).to(dtype).eval()
        return model
    
    @pytest.fixture
    def optimizer(self):
        """Create ESMOptimizer instance"""
        return ESMOptimizer()
    
    def test_apply_optimizations(self, model, optimizer):
        """Test that optimizations are applied"""
        patches = optimizer.apply(model)
        
        assert len(patches) > 0
        # Should have patches for MLP1, MLP2, and possibly LayerNorm
        patch_types = [p[0] for p in patches]
        assert "mlp1" in patch_types
        assert "mlp2" in patch_types
    
    def test_restore_optimizations(self, model, optimizer):
        """Test that optimizations can be restored"""
        # Store original forward methods
        original_mlp1 = model.encoder.layer[0].intermediate.forward
        original_mlp2 = model.encoder.layer[0].output.forward
        
        # Apply optimizations
        patches = optimizer.apply(model)
        
        # Verify patches were applied (forward methods changed)
        assert model.encoder.layer[0].intermediate.forward != original_mlp1
        assert model.encoder.layer[0].output.forward != original_mlp2
        
        # Restore
        optimizer.restore(model, patches)
        
        # Verify restoration (forward methods restored)
        assert model.encoder.layer[0].intermediate.forward == original_mlp1
        assert model.encoder.layer[0].output.forward == original_mlp2
    
    def test_optimized_forward_still_works(self, model, optimizer, device):
        """Test that optimized model still produces valid outputs"""
        # Create dummy input
        batch_size, seq_len, hidden_size = 2, 10, 320
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float16)
        
        # Apply optimizations
        patches = optimizer.apply(model)
        
        # Test MLP1 forward
        layer = model.encoder.layer[0]
        intermediate_output = layer.intermediate(hidden_states)
        assert intermediate_output.shape == (batch_size, seq_len, hidden_size * 4)
        
        # Test MLP2 forward
        output = layer.output(intermediate_output, hidden_states)
        assert output.shape == hidden_states.shape
        
        # Restore
        optimizer.restore(model, patches)
    
    def test_multiple_layers_optimized(self, model, optimizer):
        """Test that all layers are optimized"""
        num_layers = len(model.encoder.layer)
        patches = optimizer.apply(model)
        
        # Should have patches for each layer
        # Each layer should have at least MLP1 and MLP2 patches
        assert len(patches) >= num_layers * 2
    
    def test_restore_idempotent(self, model, optimizer):
        """Test that restore can be called multiple times safely"""
        patches = optimizer.apply(model)
        
        # First restore
        optimizer.restore(model, patches)
        
        # Second restore should not raise error
        optimizer.restore(model, patches)
