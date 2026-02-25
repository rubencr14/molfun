"""
LiteLLM universal backend — one interface for 100+ LLM providers.

LiteLLM proxies requests to any provider using a unified API.
The model string determines the provider:

    "gpt-4o-mini"              → OpenAI
    "claude-sonnet-4-20250514"     → Anthropic
    "ollama/llama3.1:70b"      → Ollama local
    "together_ai/meta-llama/..." → Together AI
    "groq/llama-3.1-70b..."   → Groq

Usage::

    # OpenAI via LiteLLM
    llm = LiteLLMBackend(model="gpt-4o-mini")

    # Ollama via LiteLLM
    llm = LiteLLMBackend(model="ollama/llama3.1:70b")

    # Claude via LiteLLM
    llm = LiteLLMBackend(model="claude-sonnet-4-20250514")

See https://docs.litellm.ai/docs/providers for full provider list.
"""

from __future__ import annotations
from typing import Optional
import json

from molfun.agents.llm.base import BaseLLM, LLMResponse, ToolCall


class LiteLLMBackend(BaseLLM):
    """
    Universal LLM backend via LiteLLM.

    Automatically routes to the correct provider based on the model
    string prefix. API keys are read from environment variables
    (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        api_base: Optional[str] = None,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.api_base = api_base

        try:
            import litellm
            self._litellm = litellm
        except ImportError:
            raise ImportError("litellm package required: pip install litellm")

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if self.api_base:
            kwargs["api_base"] = self.api_base

        response = self._litellm.completion(**kwargs)

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
            self._track_usage(usage)

        choice = response.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id or f"litellm_{len(tool_calls)}",
                    name=tc.function.name,
                    arguments=args,
                ))

        return LLMResponse(
            text=msg.content,
            tool_calls=tool_calls,
            usage=usage,
        )
