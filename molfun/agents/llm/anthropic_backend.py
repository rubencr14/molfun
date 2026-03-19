"""
Anthropic (Claude) backend with tool calling.

Usage::

    llm = AnthropicBackend(model="claude-sonnet-4-20250514")
    llm = AnthropicBackend(model="claude-3-5-haiku-20241022")  # cheaper
"""

from __future__ import annotations
from typing import Optional
import json
import os

from molfun.agents.llm.base import BaseLLM, LLMResponse, ToolCall


def _convert_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool schemas to Anthropic format."""
    converted = []
    for tool in tools:
        fn = tool.get("function", {})
        converted.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted


class AnthropicBackend(BaseLLM):
    """Claude API with native tool calling."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> LLMResponse:
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            elif m["role"] == "tool":
                chat_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": m["content"],
                    }],
                })
            elif m["role"] == "assistant" and m.get("tool_calls"):
                content = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"])
                               if isinstance(tc["function"]["arguments"], str)
                               else tc["function"]["arguments"],
                    })
                chat_messages.append({"role": "assistant", "content": content})
            else:
                chat_messages.append({
                    "role": m["role"],
                    "content": m["content"],
                })

        kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if tools:
            kwargs["tools"] = _convert_tools_to_anthropic(tools)

        response = self._client.messages.create(**kwargs)

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }
        self._track_usage(usage)

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            usage=usage,
        )
