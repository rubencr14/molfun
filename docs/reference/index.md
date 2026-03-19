# API Reference

Complete API reference for the Molfun library. Each page documents the public
classes, functions, and registries exposed by a subpackage.

## Core API

| Page | Description |
|------|-------------|
| [MolfunStructureModel](api/model.md) | Main facade for training, prediction, export, and hub operations |
| [Predict Functions](api/predict.md) | High-level `predict_structure`, `predict_properties`, `predict_affinity` |
| [PEFT](api/peft.md) | Parameter-Efficient Fine-Tuning utilities (LoRA, IA3) |

## Training

| Page | Description |
|------|-------------|
| [Strategies](training/strategies.md) | Fine-tuning strategies: Full, HeadOnly, LoRA, Partial |
| [Distributed](training/distributed.md) | DDP and FSDP distributed training support |

## Modular Architecture

| Page | Description |
|------|-------------|
| [Registry](modules/registry.md) | Generic `ModuleRegistry` plugin system |
| [Attention](modules/attention.md) | Attention implementations: Flash, Standard, Linear, Gated |
| [Blocks](modules/blocks.md) | Trunk blocks: Pairformer, Evoformer, SimpleTransformer |
| [Structure Modules](modules/structure-modules.md) | IPA and Diffusion structure modules |
| [Embedders](modules/embedders.md) | Input and ESM embedders |
| [ModelBuilder](modules/builder.md) | Programmatic model assembly |
| [ModuleSwapper](modules/swapper.md) | Hot-swap modules in a live model |

## Data

| Page | Description |
|------|-------------|
| [Fetchers](data/fetchers.md) | PDBFetcher, AffinityFetcher, MSAProvider |
| [Datasets](data/datasets.md) | StructureDataset, AffinityDataset, StreamingStructureDataset |
| [Splits](data/splits.md) | DataSplitter: random, temporal, identity |
| [Parsers](data/parsers.md) | PDB, mmCIF, A3M, FASTA, SDF, Mol2 parsers |
| [Collections](data/collections.md) | Curated dataset collections |

## Losses, Tracking, Export & CLI

| Page | Description |
|------|-------------|
| [Losses](losses.md) | Loss registry and built-in loss functions |
| [Tracking](tracking.md) | Experiment trackers: W&B, Comet, MLflow, Langfuse, HuggingFace, Console |
| [Export](export.md) | ONNX and TorchScript export utilities |
| [CLI](cli.md) | Command-line interface reference |
