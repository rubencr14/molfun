"""
Abstract LLM backend with tool/function calling support.

All backends normalize to the same response format so the agent
loop doesn't care whether it's talking to GPT-4o, Claude, Llama
on Ollama, or a local model via LM Studio.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolCall:
    """A single tool/function call requested by the LLM."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Normalized response from any LLM backend."""
    text: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLM(ABC):
    """
    Any LLM that supports tool/function calling.

    Subclasses implement ``chat()`` for their specific API.
    The agent loop only talks to this interface.
    """

    def __init__(self, model: str, temperature: float = 0.3, max_tokens: int = 4096):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> LLMResponse:
        """
        Send messages + tool schemas, get back a response.

        Args:
            messages: OpenAI-format message list
                [{"role": "system", "content": "..."}, ...]
            tools: OpenAI-format tool schemas (optional)

        Returns:
            LLMResponse with text and/or tool_calls.
        """
        ...

    def _track_usage(self, usage: dict) -> None:
        self.total_input_tokens += usage.get("prompt_tokens", 0)
        self.total_output_tokens += usage.get("completion_tokens", 0)

    @property
    def usage_summary(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"
