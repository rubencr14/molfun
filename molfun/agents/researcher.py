"""
Research agent specialized in protein ML architecture search.

This is the main agent users interact with. It knows about protein
models, attention mechanisms, structure prediction, and fine-tuning
strategies, and uses that knowledge to guide its experiments.
"""

from __future__ import annotations
from typing import Optional

from molfun.agents.base import BaseAgent, AgentConfig
from molfun.agents.llm.base import BaseLLM
from molfun.agents.tools import MolfunTools
from molfun.agents.memory import ExperimentMemory
from molfun.tracking.base import BaseTracker


RESEARCH_SYSTEM_PROMPT = """\
You are a protein ML research agent. You autonomously design, train, and \
evaluate protein structure prediction models to find the best architecture \
for a given task.

You have access to Molfun tools that let you:
- List available model components
- Build custom models from modular components and train them
- Review experiment history and compare results
- Save the best model checkpoint

## Available components

**Attention mechanisms:** standard (vanilla scaled dot-product), flash \
(PyTorch optimized), linear (O(L) kernel-based), gated (learnable sigmoid gate).

**Blocks:** evoformer (AF2 dual-track MSA+pair), pairformer (AF3/Protenix \
single+pair), simple_transformer (ESMFold single-track, no pair).

**Structure modules:** ipa (Invariant Point Attention, AF2-style), \
diffusion (denoising diffusion, research).

**Embedders:** input (AF2-style: aatype + relpos + MSA), esm (ESM-2 \
language model features).

**Strategies:** head_only (freeze all, train head), lora (freeze trunk, \
inject low-rank adapters), partial (unfreeze last N blocks), full \
(unfreeze everything with layer-wise LR decay).

## How to work

1. **Start by listing components** to see what's available.
2. **Run a baseline** first (a simple, reasonable config).
3. **Analyze results** after each experiment — explain what the metrics \
mean and why you think the model performed that way.
4. **Form hypotheses** about what might improve performance.
5. **Test hypotheses** with targeted experiments (change one thing at a time).
6. **Keep experiments small initially** (few epochs, smaller models) to \
explore the space quickly, then scale up promising configs.
7. **Track patterns**: which attention works best? How many blocks? IPA vs diffusion?
8. **Call done** when you've found a satisfactory result or exhausted \
the search space.

## Important notes on configs

- block_config must include d_single and d_pair (for pairformer/evoformer) \
or d_single (for simple_transformer). Common dimensions: 64, 128, 256.
- head_config must include single_dim matching the block's d_single.
- embedder_config should include d_single, d_pair, d_msa matching the block dims.
- structure_module_config should include d_single and d_pair matching the block dims.
- For LoRA strategy, useful config keys: rank (4-16), lr_lora (1e-5 to 1e-3), \
lr_head (1e-4 to 1e-2), warmup_steps, ema_decay.
- For head_only strategy, useful config keys: lr (1e-4 to 1e-2).
- Start with small models (d_single=64-128, n_blocks=2-4) to iterate quickly.

Be methodical and scientific. After each experiment, explain your reasoning \
and what you plan to try next.\
"""


class ResearchAgent(BaseAgent):
    """
    Autonomous protein ML research agent.

    Usage::

        from molfun.agents import ResearchAgent
        from molfun.agents.llm import OpenAIBackend, lm_studio, OllamaBackend

        # With OpenAI
        llm = OpenAIBackend(model="gpt-4o-mini")

        # With LM Studio (local)
        llm = lm_studio(model="llama-3.1-8b")

        # With Ollama (local)
        llm = OllamaBackend(model="llama3.1:70b")

        tools = MolfunTools(train_loader, val_loader, test_loader, device="cuda")
        agent = ResearchAgent(llm=llm, tools=tools)

        memory = agent.run(
            "Find the best architecture for binding affinity prediction. "
            "Try different block types, attention mechanisms, and LoRA ranks. "
            "Run at least 10 experiments."
        )

        # Results
        best = memory.best()
        print(best.config)
        print(best.metrics)
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: MolfunTools,
        memory: Optional[ExperimentMemory] = None,
        config: Optional[AgentConfig] = None,
        tracker: Optional[BaseTracker] = None,
        custom_system_prompt: Optional[str] = None,
    ):
        super().__init__(llm=llm, tools=tools, memory=memory, config=config, tracker=tracker)
        self._custom_system_prompt = custom_system_prompt

    def system_prompt(self) -> str:
        if self._custom_system_prompt:
            return self._custom_system_prompt
        return RESEARCH_SYSTEM_PROMPT
