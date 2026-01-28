# Contributing

This repository is a performance engineering sandbox for Triton kernels, model patching, and reproducible inference benchmarks across protein and structure ML models (e.g., ESM, AlphaFold-like stacks, Protenix, DiffDock). Contributions are welcome, but performance work is easy to regress unintentionally. Please follow the guidelines below so changes remain measurable, reproducible, and reviewable.

## Scope of Contributions

We accept contributions in these areas:

- Triton kernels (new fused primitives, improvements to existing kernels, autotuning/config updates)
- Model integration and patching utilities (clean, reversible patching of forward paths)
- Benchmarks and harness improvements (timing stability, new representative test cases, reporting)
- Correctness and validation tooling (numerical comparisons, tolerance guidelines, diagnostics)
- Documentation (kernel design notes, integration notes, benchmarking methodology)

If you are proposing a large change, open an issue first describing:
- what you want to speed up
- target models and shapes
- expected impact and why
- how you will measure and validate

## Development Principles

1. Measure first. Do not assume a kernel is faster without a benchmark and (ideally) a profile.
2. Keep patches reversible. Any runtime patching must have a clean unpatch path.
3. Prefer minimal behavioral changes. We optimize inference paths; avoid changing model semantics.
4. Correctness matters. Performance improvements must include output-difference checks.
5. Keep kernels readable. Favor clear pointer arithmetic, masking, and tile definitions over clever hacks.

## Repository Conventions

### Kernel code

- Place Triton kernels in `src/kernels/`.
- Provide a small Python wrapper per kernel that validates shapes/dtypes and handles reshaping.
- Accumulate in fp32 where it improves stability (mean/var, GEMM accumulation) unless justified.
- Support fp16 and bf16 where reasonable. If a kernel is fp16-only, document it clearly.
- Use masks for boundary handling and avoid out-of-bounds loads/stores.
- If using autotune, keep the initial config set small and safe; expand only with evidence.

### Patching code

- Patches should be narrowly scoped and avoid global side effects.
- Store original callables and restore them exactly in `unpatch_*`.
- Do not reallocate or concatenate weights inside the hot forward path.
  - If you need packed weights (e.g., fused QKV), create them once and cache them on the module.

### Benchmarks

- Benchmarks live in `src/benchmarks/`.
- Use deterministic synthetic inputs unless there is a specific need for real data.
- Use CUDA events for timing and include warmup iterations.
- Always report:
  - baseline time
  - patched time
  - speedup
  - tokens/s (or another throughput metric)
  - max/mean abs diff (or other relevant error metrics)

## How to Submit a Change

1. Fork the repository and create a feature branch.
2. Keep commits focused (one topic per PR where possible).
3. Include:
   - a clear description of the change
   - target workloads and shapes
   - benchmark results before/after
   - correctness check results
4. Open a pull request.

## Required Evidence for Performance PRs

For any change claiming performance improvement, include:

- A benchmark command (or script) and exact settings:
  - model id / architecture
  - batch size, sequence length (or other shape parameters)
  - dtype (fp16/bf16)
  - iterations, warmup
- Before/after numbers:
  - ms/iter and tokens/s
  - speedup
- Correctness deltas:
  - max abs diff, mean abs diff (or similar)
- If the change is non-trivial, include a profile:
  - Nsight Systems (`nsys`) trace summary is usually sufficient

Example reporting format:

- Hardware: GPU model, driver/CUDA version
- Software: PyTorch version, Triton version, Transformers version
- Benchmark: script name, cases
- Baseline: ms/iter, tokens/s
- Patched: ms/iter, tokens/s
- Diff: max abs, mean abs
- Notes: what changed, why it should be faster

## Correctness Guidelines

This repository focuses on inference. Fused kernels may introduce small numerical differences due to:
- fp16/bf16 arithmetic
- different accumulation order
- fused epilogues changing rounding behavior
- alternative activation approximations

When adding or modifying kernels:
- Keep default behavior consistent with reference implementations unless explicitly documented.
- If you introduce an approximation (e.g., tanh GELU), document it and update tests/bench tolerances.
- Prefer comparing the final `last_hidden_state` (or equivalent output) for integration benchmarks.

## Style and Quality

- Use clear naming: `fused_*_kernel` for Triton kernels, `*_triton(...)` for wrappers.
- Add concise comments explaining tiling, pointer arithmetic, and masking.
- Avoid unnecessary dependencies.
- Keep Python code compatible with typical PyTorch/Transformers usage patterns.

## Reporting Bugs

If you find a correctness issue or crash:
- Include the smallest reproducible snippet or the benchmark script and exact parameters.
- Include hardware and software versions.
- If it is a Triton compilation issue, include the error log and the kernel configuration if available.

## License

By contributing, you agree that your contributions will be licensed under the repository’s Apache License 2.0.