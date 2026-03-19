# Contributing

Molfun is a framework for fine-tuning molecular ML models, building modular protein architectures, and GPU-accelerated molecular analysis. Contributions are welcome across all areas of the project.

## Scope of Contributions

We accept contributions in these areas:

- **Model adapters** — new backend integrations (ESMFold, Protenix, docking models)
- **Modular components** — new attention mechanisms, blocks, structure modules, embedders
- **Training strategies** — new fine-tuning approaches, schedulers, regularization techniques
- **Data pipeline** — new data sources, featurizers, splitting strategies
- **GPU kernels** — Triton kernels for geometric primitives and model internals
- **Benchmarks** — training benchmarks, kernel performance, architecture comparisons
- **Tests** — unit tests, integration tests, GPU validation
- **Documentation** — guides, tutorials, API documentation

If you are proposing a large change, open an issue first describing:
- what you want to add or change
- the motivation and expected impact
- how you will validate it

## Development Principles

1. **Interfaces first.** New components should implement the relevant abstract base class (`BaseAttention`, `BaseBlock`, `BaseStructureModule`, `BaseEmbedder`, `BaseAdapter`) and register themselves in the appropriate registry.
2. **Keep things swappable.** Design components so they can be substituted at runtime without modifying other parts of the model.
3. **Measure first.** Performance claims should be backed by benchmarks. Architecture comparisons should include controlled experiments.
4. **Correctness matters.** Include tests for new components — at minimum shape checks and gradient flow validation.
5. **Backward compatible.** Changes to existing interfaces should not break existing adapters, strategies, or modules.

## Repository Conventions

### Modular components

- Place new attention implementations in `molfun/modules/attention/`.
- Place new blocks in `molfun/modules/blocks/`.
- Place new structure modules in `molfun/modules/structure_module/`.
- Place new embedders in `molfun/modules/embedders/`.
- Register every new component using `@REGISTRY.register("name")`.
- Implement the full abstract interface (all abstract methods and properties).
- Add tests in `tests/modules/`.

### Model adapters

- Place new adapters in `molfun/adapters/`.
- Implement `BaseAdapter`: `forward()`, `freeze_trunk()`, `unfreeze_trunk()`, `get_trunk_blocks()`, `peft_target_module`, `default_peft_targets`, and optionally `get_structure_module()`, `get_input_embedder()`.
- Register the adapter so it's accessible via `MolfunStructureModel("name", ...)`.

### Training strategies

- Place new strategies in `molfun/training/`.
- Extend `FinetuneStrategy` from `molfun/training/base.py`.
- Include proper parameter group setup with differential learning rates.

### Kernel code

- Place Triton kernels in `molfun/kernels/`.
- Provide a Python wrapper per kernel that validates shapes/dtypes.
- Accumulate in fp32 where it improves stability.
- Use masks for boundary handling.

### Benchmarks

- Benchmarks live in `molfun/benchmarks/`.
- Use deterministic synthetic inputs unless there is a specific need for real data.
- Use CUDA events for timing and include warmup iterations.
- Report baseline, optimized, speedup, and correctness metrics.

## How to Submit a Change

1. Fork the repository and create a feature branch.
2. Keep commits focused (one topic per PR where possible).
3. Include:
   - a clear description of the change
   - tests covering the new functionality
   - benchmark results if claiming performance improvement
4. Open a pull request.

## Required Evidence for Performance PRs

For any change claiming performance improvement, include:

- Benchmark command/script and exact settings (model, shapes, dtype, iterations)
- Before/after numbers (ms/iter, speedup)
- Correctness deltas (max abs diff, mean abs diff)
- Hardware and software versions (GPU, CUDA, PyTorch, Triton)

## Style and Quality

- Follow existing naming conventions in each subpackage.
- Add docstrings to public classes and methods.
- Avoid unnecessary dependencies — core functionality should work with just PyTorch.
- Optional backends (OpenFold, ESM, Triton, HuggingFace PEFT) are guarded by try/except imports.

## Reporting Bugs

If you find a correctness issue or crash:
- Include the smallest reproducible snippet.
- Include hardware and software versions.
- Include the full error traceback.

## License

By contributing, you agree that your contributions will be licensed under the repository's Apache License 2.0.
