# Triton Kernel Patching for Protein and Structure Models Inference

This repository is an experimental performance engineering sandbox focused on accelerating inference workloads in protein and molecular-structure machine learning pipelines using custom Triton kernels and lightweight runtime patching. While ESM-style protein language models are a natural starting point for kernel fusion work, the scope of this repo is broader: it is intended to become a shared kernel and benchmarking toolkit that can be applied across multiple model families commonly used in computational biology workflows, including protein language models (e.g., ESM), structure prediction systems (e.g., AlphaFold-like stacks, Protenix), and protein–ligand docking models (e.g., DiffDock).

The motivation is practical. These models are built from repeating blocks—linear projections, activations, residual adds, normalization layers, and attention-like subgraphs—that are typically executed as many separate GPU kernels. 

Even when individual GEMMs run close to peak performance, end-to-end latency is often dominated by global memory traffic (HBM/VRAM round trips), synchronization points, and kernel launch overhead from small or medium “glue” ops (bias adds, GELU, residual adds, layer norm, reshapes, casts). This repository aims to reduce that overhead by fusing common patterns into fewer kernels so that intermediate values are produced and consumed with less materialization to global memory.

The project combines two complementary pieces. First, it provides a growing library of Triton kernels that implement fused primitives commonly encountered in these architectures, such as `Linear + Bias + Activation` and `Linear + Bias + Residual Add`, typically with fp32 accumulation for numerical stability and fp16/bf16 I/O for throughput. Second, it provides patching and integration patterns that let you drop these kernels into existing PyTorch modules without rewriting entire codebases: specific `forward` methods can be replaced at runtime to call fused kernels, enabling rapid A/B testing against baseline implementations.

A central focus of the repository is measurement-driven iteration. It includes reproducible benchmarks that isolate performance as a function of tensor shapes, which is what dictates GPU runtime. For protein language models, this means benchmarking across batch size and sequence length. For structure prediction and docking models, this means constructing representative synthetic shapes for the key compute regions (e.g., trunk blocks, pair representations, attention-like interactions, projection-heavy MLP subgraphs). 

Each benchmark is designed to support a tight optimization loop: warmups to trigger Triton JIT compilation and autotuning, CUDA-event timing for stable measurements, correctness checks against the baseline outputs, and optional profiling with Nsight Systems to identify true bottlenecks.

Importantly, the repository is not limited to one model’s internal details. Instead, it treats these systems as a set of recurring computational motifs. 

ESM-style encoders highlight MLP and normalization fusion opportunities; AlphaFold-like models introduce large pair/triangle update blocks and heavy normalization/residual patterns; docking models often combine attention-like modules, geometric processing, and projection-heavy sub-networks. 

The repo’s goal is to capture the reusable kernels and patching strategies that apply across these domains, so improvements made for one model family can transfer to others with minimal integration work.

In short, this repository is intended as a pragmatic “kernel lab” for modern protein and structure ML inference: implement a fused kernel, patch it into a real model, benchmark on representative shapes, validate numerical behavior, profile bottlenecks, and iterate. Over time, it becomes a shared optimization layer for multiple pipelines—ESM, AlphaFold-like systems, Protenix, DiffDock—providing reusable fused primitives and a repeatable workflow for systematically improving GPU utilization and end-to-end latency.
