"""
Ollama backend for fully local LLM inference.

Requires Ollama running locally (https://ollama.com).

Usage::

    llm = OllamaBackend(model="llama3.1:70b")
    llm = OllamaBackend(model="qwen2.5:72b")
    llm = OllamaBackend(model="mistral:7b", host="http://192.168.1.50:11434")
"""

from __future__ import annotations
from typing import Optional
import json

from molfun.agents.llm.base import BaseLLM, LLMResponse, ToolCall


def _convert_tools_to_ollama(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool schemas to Ollama format."""
    converted = []
    for tool in tools:
        fn = tool.get("function", {})
        converted.append({
            "type": "function",
            "function": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            },
        })
    return converted


class OllamaBackend(BaseLLM):
    """
    Ollama local inference with tool calling.

    Tool calling requires Ollama >= 0.4 and a model that supports it
    (llama3.1, qwen2.5, mistral, etc.).
    """

    def __init__(
        self,
        model: str = "llama3.1",
        host: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.host = host

        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama package required: pip install ollama\n"
                "Also ensure Ollama is running: https://ollama.com"
            )

        kwargs = {}
        if host:
            kwargs["host"] = host
        self._client = ollama.Client(**kwargs)

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> LLMResponse:
        clean_messages = []
        for m in messages:
            if m["role"] == "tool":
                clean_messages.append({
                    "role": "tool",
                    "content": m["content"],
                })
            else:
                clean_messages.append({
                    "role": m["role"],
                    "content": m.get("content", ""),
                })

        kwargs = {
            "model": self.model,
            "messages": clean_messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if tools:
            kwargs["tools"] = _convert_tools_to_ollama(tools)

        response = self._client.chat(**kwargs)

        usage = {}
        if hasattr(response, "prompt_eval_count"):
            usage["prompt_tokens"] = getattr(response, "prompt_eval_count", 0)
            usage["completion_tokens"] = getattr(response, "eval_count", 0)
            self._track_usage(usage)

        msg = response.get("message", response) if isinstance(response, dict) else response.message

        tool_calls = []
        msg_tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        if msg_tool_calls:
            for i, tc in enumerate(msg_tool_calls):
                fn = tc.get("function", tc) if isinstance(tc, dict) else tc.function
                fn_name = fn.get("name", "") if isinstance(fn, dict) else fn.name
                fn_args = fn.get("arguments", {}) if isinstance(fn, dict) else fn.arguments
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except json.JSONDecodeError:
                        fn_args = {}
                tool_calls.append(ToolCall(
                    id=f"ollama_{i}",
                    name=fn_name,
                    arguments=fn_args,
                ))

        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

        return LLMResponse(
            text=content if content else None,
            tool_calls=tool_calls,
            usage=usage,
        )
