"""
molfun.agents — LLM-powered autonomous research agents.

Agents use a language model as a "brain" to autonomously design,
train, and evaluate protein ML models using Molfun's modular
architecture. They can run for hours unattended on a GPU server,
systematically exploring the architecture and hyperparameter space.

Quick start::

    from molfun.agents import ResearchAgent
    from molfun.agents.llm import OpenAIBackend  # or lm_studio, OllamaBackend

    llm = OpenAIBackend(model="gpt-4o-mini")
    tools = MolfunTools(train_loader, val_loader, device="cuda")
    agent = ResearchAgent(llm=llm, tools=tools)

    memory = agent.run("Find the best architecture for binding affinity.")
    print(memory.best().config)

Supported LLM backends:
    - OpenAI API (GPT-4o, GPT-4o-mini)
    - LM Studio (any local model, OpenAI-compatible)
    - Ollama (llama3.1, qwen2.5, mistral, etc.)
    - Anthropic (Claude)
    - LiteLLM (universal proxy for 100+ providers)
    - vLLM (local OpenAI-compatible server)
"""

from molfun.agents.experiment import ExperimentConfig, Experiment
from molfun.agents.memory import ExperimentMemory
from molfun.agents.tools import MolfunTools
from molfun.agents.base import BaseAgent, AgentConfig
from molfun.agents.researcher import ResearchAgent

__all__ = [
    "ExperimentConfig",
    "Experiment",
    "ExperimentMemory",
    "MolfunTools",
    "BaseAgent",
    "AgentConfig",
    "ResearchAgent",
]
