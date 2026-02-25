"""
OpenAI-compatible backend.

Works with:
- OpenAI API (GPT-4o, GPT-4o-mini, o1, etc.)
- LM Studio (local, runs at http://localhost:1234/v1)
- vLLM (local, OpenAI-compatible server)
- Any OpenAI-compatible endpoint (Together, Groq, etc.)

Usage::

    # OpenAI API
    llm = OpenAIBackend(model="gpt-4o-mini", api_key="sk-...")

    # LM Studio (local)
    llm = OpenAIBackend(
        model="llama-3.1-8b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",  # LM Studio ignores the key
    )

    # vLLM local server
    llm = OpenAIBackend(
        model="Qwen/Qwen2.5-72B-Instruct",
        base_url="http://localhost:8000/v1",
        api_key="dummy",
    )
"""

from __future__ import annotations
from typing import Optional
import json
import os

from molfun.agents.llm.base import BaseLLM, LLMResponse, ToolCall


class OpenAIBackend(BaseLLM):
    """
    OpenAI-compatible chat completions with tool calling.

    Set ``base_url`` to point at LM Studio, vLLM, or any
    OpenAI-compatible local server.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required: pip install openai\n"
                "This backend works with OpenAI API, LM Studio, vLLM, etc."
            )
        kwargs = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(api_key=self.api_key, **kwargs)

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
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)

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
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        return LLMResponse(
            text=msg.content,
            tool_calls=tool_calls,
            usage=usage,
        )


def lm_studio(
    model: str = "local-model",
    port: int = 1234,
    **kwargs,
) -> OpenAIBackend:
    """Convenience constructor for LM Studio."""
    return OpenAIBackend(
        model=model,
        base_url=f"http://localhost:{port}/v1",
        api_key="lm-studio",
        **kwargs,
    )


def vllm_local(
    model: str = "local-model",
    port: int = 8000,
    **kwargs,
) -> OpenAIBackend:
    """Convenience constructor for vLLM local server."""
    return OpenAIBackend(
        model=model,
        base_url=f"http://localhost:{port}/v1",
        api_key="dummy",
        **kwargs,
    )
