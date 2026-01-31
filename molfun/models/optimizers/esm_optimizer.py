"""
ESM-specific optimizations using Triton kernels.
"""

from __future__ import annotations
from typing import List
import types

import torch
from transformers import EsmModel

from molfun.kernels.models.fused_linear_gelu_triton import fused_linear_gelu_triton
from molfun.kernels.models.fused_linear_bias_residual_triton import fused_linear_bias_residual_triton
from molfun.kernels.models.layernorm_triton import layernorm_triton

from molfun.models.base_model import ModelOptimizer


class ESMOptimizer(ModelOptimizer):
    """Applies Triton kernel optimizations to ESM models"""
    
    def apply(self, model: EsmModel) -> List:
        """Apply optimizations to ESM model"""
        patches_restore = []
        
        for layer in model.encoder.layer:
            # Patch MLP1: fused_linear_gelu_triton
            intermediate = layer.intermediate
            original_mlp1 = intermediate.forward
            
            def make_mlp1_patch(intermediate_module):
                def patched_mlp1(self_intermediate, hidden_states):
                    w = intermediate_module.dense.weight
                    b = intermediate_module.dense.bias
                    return fused_linear_gelu_triton(hidden_states, w, b)
                return patched_mlp1
            
            intermediate.forward = types.MethodType(make_mlp1_patch(intermediate), intermediate)
            patches_restore.append(("mlp1", intermediate, original_mlp1))
            
            # Patch MLP2: fused_linear_bias_residual_triton
            output = layer.output
            original_mlp2 = output.forward
            
            def make_mlp2_patch(output_module):
                def patched_mlp2(self_output, hidden_states, input_tensor):
                    w = output_module.dense.weight
                    b = output_module.dense.bias
                    y = fused_linear_bias_residual_triton(hidden_states, w, b, input_tensor)
                    # Preserve HF semantics (dropout + LayerNorm if present)
                    if hasattr(output_module, "dropout") and output_module.dropout is not None:
                        y = output_module.dropout(y)
                    if hasattr(output_module, "LayerNorm") and output_module.LayerNorm is not None:
                        y = output_module.LayerNorm(y)
                    return y
                return patched_mlp2
            
            output.forward = types.MethodType(make_mlp2_patch(output), output)
            patches_restore.append(("mlp2", output, original_mlp2))
            
            # Patch LayerNorm: layernorm_triton
            if hasattr(layer, "layer_norm") and layer.layer_norm is not None:
                norm_module = layer.layer_norm
                original_norm = norm_module.forward
                
                def make_norm_patch(norm_mod):
                    def patched_norm(self_norm, hidden_states):
                        gamma = norm_mod.weight
                        beta = norm_mod.bias
                        eps = norm_mod.eps
                        return layernorm_triton(hidden_states, gamma, beta, eps)
                    return patched_norm
                
                norm_module.forward = types.MethodType(make_norm_patch(norm_module), norm_module)
                patches_restore.append(("norm", norm_module, original_norm))
        
        return patches_restore
    
    def restore(self, model: EsmModel, patches: List):
        """Restore original model state"""
        for patch_type, module, original in patches:
            if hasattr(module, 'forward'):
                module.forward = original
