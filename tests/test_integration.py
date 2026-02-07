"""
Integration tests for BioLanguageModel with optimizations.
"""

import pytest
import torch

from molfun import BioLanguageModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestIntegration:
    """Integration tests for full model pipeline"""
    
    def test_optimized_vs_unoptimized_outputs(self, device, dtype, single_sequence):
        """Test that optimized and unoptimized models produce similar outputs"""
        # Unoptimized model
        model_unopt = BioLanguageModel.from_pretrained(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            use_optimizations=False,
        )
        model_unopt.eval()
        output_unopt = model_unopt.infer(single_sequence)
        
        # Optimized model
        model_opt = BioLanguageModel.from_pretrained(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            use_optimizations=True,
        )
        model_opt.eval()
        output_opt = model_opt.infer(single_sequence)
        
        # Outputs should have same shape
        assert output_unopt.shape == output_opt.shape
        
        # Check numerical similarity (allowing for small differences due to optimizations)
        max_diff = (output_unopt - output_opt).abs().max().item()
        mean_diff = (output_unopt - output_opt).abs().mean().item()
        
        # Differences should be small (optimizations should preserve semantics)
        # Note: Exact match not expected due to fused operations, but should be close
        assert max_diff < 1e-2  # Reasonable tolerance for fp16
        assert mean_diff < 1e-3
    
    def test_batch_inference_consistency(self, device, dtype, sample_sequences):
        """Test that batch inference is consistent with individual inference"""
        model = BioLanguageModel.from_pretrained(
            model_name="ESM-2-8M",
            dtype=dtype,
            device=device,
            use_optimizations=False,
        )
        model.eval()
        
        # Batch inference
        batch_output = model.infer(sample_sequences)
        
        # Individual inference
        individual_outputs = [model.infer(seq) for seq in sample_sequences]
        
        # Compare shapes and values
        assert batch_output.shape[0] == len(sample_sequences)
        
        # For float16, use more lenient tolerance due to numerical precision
        # Batch processing can introduce small differences due to padding and parallel execution
        atol = 1e-2 if dtype == torch.float16 else 1e-5
        
        for i, individual_output in enumerate(individual_outputs):
            # Batch output at index i should match individual output
            # (allowing for padding differences)
            batch_seq = batch_output[i]
            ind_seq = individual_output[0]  # Remove batch dim
            
            # Compare non-padded parts - use minimum length to avoid padding issues
            min_len = min(batch_seq.shape[0], ind_seq.shape[0])
            
            # For float16, small numerical differences are expected due to:
            # - Different padding behavior in batch vs individual
            # - Floating point precision limits
            # - Potential differences in computation order
            max_diff = (batch_seq[:min_len] - ind_seq[:min_len]).abs().max().item()
            mean_diff = (batch_seq[:min_len] - ind_seq[:min_len]).abs().mean().item()
            
            # Allow larger differences for float16 due to precision limits
            # The outputs should be reasonably close but exact match not required
            assert max_diff < atol, f"Max difference {max_diff:.6f} exceeds tolerance {atol}"
            assert mean_diff < atol * 0.1, f"Mean difference {mean_diff:.6f} exceeds tolerance"
    
    def test_different_models_same_interface(self, device, dtype, single_sequence):
        """Test that different models use the same interface"""
        models = [
            BioLanguageModel.from_pretrained(
                model_name="ESM-2-8M",
                dtype=dtype,
                device=device,
                use_optimizations=False,
            ),
            BioLanguageModel.from_pretrained(
                model_name="ESM-2-150M",
                dtype=dtype,
                device=device,
                use_optimizations=False,
            ),
        ]
        
        for model in models:
            model.eval()
            output = model.infer(single_sequence)
            
            assert isinstance(output, torch.Tensor)
            assert output.dim() == 3
            assert output.shape[0] == 1
    
    def test_model_reuse(self, device, dtype, sample_sequences):
        """Test reusing the same model for multiple inferences"""
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
        
        assert len(outputs) == len(sample_sequences)
        assert all(isinstance(out, torch.Tensor) for out in outputs)
        assert all(out.shape[0] == 1 for out in outputs)
