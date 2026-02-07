# Tests

Comprehensive test suite for Molfun models, backends, optimizers, and integration scenarios.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_bio_language_model.py

# Run with verbose output
pytest -v

# Run only fast tests (skip slow ones)
pytest -m "not slow"

# Run only CUDA tests (requires CUDA)
pytest -m cuda

# Run with coverage
pytest --cov=molfun --cov-report=html
```

## Test Structure

### `test_bio_language_model.py`

Tests the high-level `BioLanguageModel` API, which serves as the main interface for users. This test suite verifies that models can be instantiated correctly for both supported ESM variants (ESM-2-8M and ESM-2-150M), ensuring proper initialization with various configuration options including data types, device placement, and optimization settings. The tests validate that the `eval()` method correctly sets models into evaluation mode and returns the model instance for method chaining. Inference functionality is thoroughly tested with both single sequences and batched inputs, confirming that outputs have the correct tensor shapes, data types, and device placement. The suite also verifies that maximum sequence length constraints are properly enforced during tokenization and inference. Additionally, tests confirm that optimization features can be toggled on and off, and that when enabled, optimizations are correctly applied and tracked. Output consistency is validated across different inference calls, ensuring that hidden dimensions remain stable and data types match the configured model precision.

### `test_esm_backend.py`

Tests the `ESMBackend` implementation, which provides the low-level interface for ESM model operations. These tests verify that backend instances can be created for both ESM-2-8M and ESM-2-150M models, with proper configuration of data types and device placement. The test suite validates lazy loading behavior, ensuring that models and tokenizers are only loaded when first accessed through their respective properties, which improves startup performance. Evaluation mode functionality is tested to confirm that models are correctly set to non-training state. Inference operations are validated for both single sequences and batches, checking that outputs maintain correct tensor dimensions and are placed on the expected device. The tests also verify that maximum sequence length constraints are properly applied during tokenization. Additionally, the backend is tested for its ability to accept direct HuggingFace model identifiers as a fallback mechanism, ensuring flexibility in model specification.

### `test_esm_optimizer.py`

Tests the `ESMOptimizer` class, which applies Triton kernel optimizations to ESM models. The test suite verifies that optimizations can be successfully applied to model layers, specifically checking that MLP1 (intermediate dense layer with GELU activation) and MLP2 (output dense layer with residual connection) are patched with fused Triton kernels. Tests confirm that the optimization process returns a list of patch restoration functions that can be used to revert changes. The restoration functionality is validated to ensure that original forward methods can be correctly restored, returning models to their unoptimized state. Functional correctness is verified by testing that optimized models still produce valid outputs with correct tensor shapes, even after patches are applied. The test suite also confirms that optimizations are applied to all encoder layers in the model, not just the first layer. Additionally, tests verify that restoration operations are idempotent and can be safely called multiple times without causing errors.

### `test_integration.py`

Integration tests that validate the complete model pipeline from initialization through inference, including optimization workflows. These tests compare outputs from optimized and unoptimized models to ensure that Triton kernel optimizations preserve numerical semantics while potentially improving performance. The comparison validates that output shapes match exactly and that numerical differences remain within acceptable tolerances for the configured data type. Batch inference consistency is tested by comparing results from batched inputs against individual sequence inferences, ensuring that padding and batching logic work correctly. The test suite verifies that different model variants (ESM-2-8M and ESM-2-150M) maintain a consistent interface, allowing code to work seamlessly across model sizes. Model reuse scenarios are tested to confirm that the same model instance can be used for multiple sequential inference calls without degradation or state leakage. These integration tests provide confidence that the entire system works correctly as a cohesive unit.

## Requirements

Tests require:
- pytest
- torch
- transformers
- CUDA (for most tests)

Install test dependencies:
```bash
pip install pytest pytest-cov
```
