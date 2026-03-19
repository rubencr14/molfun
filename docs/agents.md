# Molfun Agents — Autonomous Protein ML Research

Molfun agents use a language model as a "brain" to autonomously design, build,
train, and evaluate protein ML models. The LLM reasons about results and decides
what to try next, running unattended for hours on a GPU server.

## Architecture

```
┌─────────────────────────────────────────┐
│              LLM Backend                │
│  OpenAI · LM Studio · Ollama · Claude   │
│  LiteLLM · vLLM · any OpenAI-compat    │
└───────────────┬─────────────────────────┘
                │  tool calls / results
┌───────────────▼─────────────────────────┐
│           BaseAgent loop                │
│  system prompt → LLM → execute → repeat │
│  context management · error recovery    │
└───────────────┬─────────────────────────┘
                │
    ┌───────────▼───────────┐
    │     MolfunTools       │
    │  list_components      │
    │  build_and_train      │
    │  get_experiment       │
    │  get_journal          │
    │  save_best_model      │
    │  done                 │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │     Molfun Core       │
    │  ModelBuilder         │
    │  MolfunStructureModel │
    │  FinetuneStrategy     │
    │  Module Registries    │
    └───────────────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
# Core + OpenAI backend (also works with LM Studio)
pip install molfun[agents]

# Or with Ollama
pip install molfun[agents-ollama]

# Or with Claude
pip install molfun[agents-anthropic]

# Or universal (all providers via LiteLLM)
pip install molfun[agents-litellm]
```

### 2. Choose your LLM backend

```python
# --- Cloud APIs ---

# OpenAI
from molfun.agents.llm import OpenAIBackend
llm = OpenAIBackend(model="gpt-4o-mini")

# Claude
from molfun.agents.llm import AnthropicBackend
llm = AnthropicBackend(model="claude-sonnet-4-20250514")


# --- Local models ---

# LM Studio (runs at localhost:1234, OpenAI-compatible)
from molfun.agents.llm import lm_studio
llm = lm_studio(model="llama-3.1-8b-instruct", port=1234)

# Ollama
from molfun.agents.llm import OllamaBackend
llm = OllamaBackend(model="llama3.1:70b")

# vLLM server
from molfun.agents.llm import vllm_local
llm = vllm_local(model="Qwen/Qwen2.5-72B-Instruct", port=8000)

# Any OpenAI-compatible server
from molfun.agents.llm import OpenAIBackend
llm = OpenAIBackend(
    model="my-model",
    base_url="http://my-server:8080/v1",
    api_key="optional-key",
)


# --- Universal (auto-routes by model string) ---

from molfun.agents.llm import LiteLLMBackend
llm = LiteLLMBackend(model="ollama/llama3.1:70b")     # local
llm = LiteLLMBackend(model="gpt-4o-mini")              # OpenAI
llm = LiteLLMBackend(model="claude-sonnet-4-20250514")      # Anthropic
```

### 3. Prepare your data

```python
from torch.utils.data import DataLoader

# Your protein dataset (PDB structures + affinity labels)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)
```

### 4. Run the agent

```python
from molfun.agents import ResearchAgent, MolfunTools, AgentConfig

tools = MolfunTools(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device="cuda",
)

agent = ResearchAgent(
    llm=llm,
    tools=tools,
    config=AgentConfig(
        max_steps=50,        # max LLM turns
        verbose=True,        # print progress
    ),
)

memory = agent.run(
    "Find the best architecture for binding affinity prediction. "
    "Try different block types (pairformer, evoformer, simple_transformer), "
    "attention mechanisms (standard, flash, linear, gated), "
    "and LoRA ranks (4, 8, 16). Start with small models to explore, "
    "then scale up the best configs. Run at least 10 experiments."
)
```

### 5. Inspect results

```python
# Best experiment
best = memory.best()
print(best.config.to_dict())
print(best.metrics)

# All experiments
for exp in memory.completed:
    print(exp.summary_line())

# Full summary
print(memory.summary_for_context())
```

## LM Studio Setup

[LM Studio](https://lmstudio.ai/) lets you run open-weight models locally with
an OpenAI-compatible API. This is the easiest way to run agents without API costs.

1. **Download LM Studio** from https://lmstudio.ai/
2. **Download a model** — recommended for tool calling:
   - Llama 3.1 8B Instruct (good balance of speed/quality)
   - Qwen 2.5 7B/72B Instruct (excellent tool calling)
   - Mistral 7B Instruct
3. **Start the local server** (Developer tab → Start Server)
4. **Connect from Molfun:**

```python
from molfun.agents.llm import lm_studio

# Default port is 1234
llm = lm_studio(model="llama-3.1-8b-instruct")

# Custom port
llm = lm_studio(model="qwen2.5-72b-instruct", port=5000)
```

> **Note:** Tool calling quality varies by model. Llama 3.1+ and Qwen 2.5+
> have native tool calling support and work best with Molfun agents.

## Ollama Setup

[Ollama](https://ollama.com/) is another option for local inference.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model with tool calling support
ollama pull llama3.1:70b

# The agent connects automatically
```

```python
from molfun.agents.llm import OllamaBackend

llm = OllamaBackend(model="llama3.1:70b")

# Remote Ollama server
llm = OllamaBackend(model="llama3.1:70b", host="http://gpu-server:11434")
```

## Persistent Memory

Agents can resume after crashes by persisting their experiment memory:

```python
from molfun.agents import ExperimentMemory

# Memory persists to disk
memory = ExperimentMemory(persist_path="experiments/run_001.json")

agent = ResearchAgent(llm=llm, tools=tools, memory=memory)
memory = agent.run("...")

# Later — resume from saved state
memory = ExperimentMemory.load("experiments/run_001.json")
print(f"Resumed with {memory.count} previous experiments")
print(f"Best so far: {memory.best().summary_line()}")
```

## Custom Agents

You can subclass `BaseAgent` to create domain-specific agents:

```python
from molfun.agents.base import BaseAgent

class AblationAgent(BaseAgent):
    """Agent that systematically ablates components."""

    def system_prompt(self) -> str:
        return """\
You are an ablation study agent. Given a baseline config,
systematically remove or replace one component at a time
to understand each component's contribution.

Start with the baseline, then:
1. Remove/replace the attention mechanism
2. Change the block type
3. Vary the number of blocks
4. Try different structure modules
5. Compare training strategies

Report the contribution of each component.
"""


class ScalingAgent(BaseAgent):
    """Agent that finds optimal model size."""

    def system_prompt(self) -> str:
        return """\
You are a scaling law agent. Your job is to find the optimal
model size for the given compute budget.

Start very small (d_single=32, n_blocks=1) and progressively
scale up. Track val_loss vs parameter count. Stop when
scaling yields diminishing returns.
"""
```

## Available Tools

The agent has access to these tools:

| Tool | Description |
|------|-------------|
| `list_components` | List all available attention, block, structure module, embedder, and strategy types |
| `build_and_train` | Build a custom model and train it end-to-end. Returns metrics and experiment ID |
| `get_experiment` | Get full details (config, history, metrics) for a specific experiment |
| `get_journal` | Get compressed summary of all experiments for context |
| `save_best_model` | Save a model checkpoint to disk |
| `done` | Signal that the agent has finished its search |

## How It Works

1. The agent starts with a **system prompt** defining its role and available components
2. The user provides an **objective** (natural language)
3. The agent loop:
   - Sends messages + tool schemas to the LLM
   - LLM decides which tool to call (or sends reasoning text)
   - Tool is executed (e.g., model is built and trained)
   - Result is appended to conversation
   - **Experiment memory** is updated and persisted
   - Context is **compacted** periodically to fit the LLM's context window
   - Loop repeats until `done` is called or `max_steps` is reached
4. The agent returns the `ExperimentMemory` with all findings
