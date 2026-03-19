# Architecture

Molfun is built on three principles:

**Modularity** -- Every component (attention, blocks, structure modules, embedders, losses) lives behind an abstract interface and a registry. Swap one implementation for another with a single line of code.

**Extensibility** -- Adding a new attention mechanism, training strategy, or model backend never requires modifying existing code. Register your class with a decorator and the framework picks it up.

**Scientific rigor** -- Protein structure prediction demands reproducible experiments. Molfun's training framework enforces consistent infrastructure (EMA, gradient accumulation, checkpointing, early stopping) across all fine-tuning strategies so results are comparable.

---

## Architecture guides

| Guide | What you will learn |
|-------|---------------------|
| [System Overview](overview.md) | Logical layers, subsystem map, request flow |
| [Design Patterns](patterns.md) | Registry, Strategy, Adapter, Template Method, Facade |
| [Module System](module-system.md) | 4 module families, registries, builder, swapper |
| [Training Framework](training-framework.md) | Fine-tuning strategies, training loop, checkpointing |
| [Data Pipeline](data-pipeline.md) | Sources, parsers, datasets, splits, storage |
