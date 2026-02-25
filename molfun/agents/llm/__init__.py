"""
LLM backends for Molfun agents.

All backends normalize to the same BaseLLM interface with tool calling.

Quick start — pick your backend:

    # OpenAI API
    from molfun.agents.llm import OpenAIBackend
    llm = OpenAIBackend(model="gpt-4o-mini")

    # LM Studio (local, OpenAI-compatible)
    from molfun.agents.llm import lm_studio
    llm = lm_studio(model="llama-3.1-8b", port=1234)

    # Ollama (local)
    from molfun.agents.llm import OllamaBackend
    llm = OllamaBackend(model="llama3.1:70b")

    # Claude
    from molfun.agents.llm import AnthropicBackend
    llm = AnthropicBackend(model="claude-sonnet-4-20250514")

    # Universal (auto-routes by model string)
    from molfun.agents.llm import LiteLLMBackend
    llm = LiteLLMBackend(model="ollama/llama3.1:70b")
"""

from molfun.agents.llm.base import BaseLLM, LLMResponse, ToolCall

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "ToolCall",
]


def _lazy_import(name):
    """Import backends lazily to avoid requiring all dependencies."""
    if name == "OpenAIBackend":
        from molfun.agents.llm.openai_backend import OpenAIBackend
        return OpenAIBackend
    elif name == "lm_studio":
        from molfun.agents.llm.openai_backend import lm_studio
        return lm_studio
    elif name == "vllm_local":
        from molfun.agents.llm.openai_backend import vllm_local
        return vllm_local
    elif name == "AnthropicBackend":
        from molfun.agents.llm.anthropic_backend import AnthropicBackend
        return AnthropicBackend
    elif name == "OllamaBackend":
        from molfun.agents.llm.ollama_backend import OllamaBackend
        return OllamaBackend
    elif name == "LiteLLMBackend":
        from molfun.agents.llm.litellm_backend import LiteLLMBackend
        return LiteLLMBackend
    raise AttributeError(f"module 'molfun.agents.llm' has no attribute {name!r}")


def __getattr__(name):
    return _lazy_import(name)
