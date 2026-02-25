"""Tests for the LLM abstraction layer."""

import json
import pytest

from molfun.agents.llm.base import BaseLLM, LLMResponse, ToolCall


class MockLLM(BaseLLM):
    """Mock LLM for testing the agent loop without real API calls."""

    def __init__(self, responses: list[LLMResponse]):
        super().__init__(model="mock", temperature=0.0)
        self._responses = list(responses)
        self._call_count = 0
        self.call_log: list[dict] = []

    def chat(self, messages, tools=None):
        self.call_log.append({
            "messages": messages,
            "tools": tools,
        })
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
            self._call_count += 1
            return response
        return LLMResponse(text="No more responses.")


class TestLLMResponse:
    def test_text_only(self):
        r = LLMResponse(text="Hello")
        assert not r.has_tool_calls
        assert r.text == "Hello"

    def test_with_tool_calls(self):
        r = LLMResponse(tool_calls=[
            ToolCall(id="tc_1", name="list_components", arguments={}),
        ])
        assert r.has_tool_calls
        assert r.tool_calls[0].name == "list_components"


class TestBaseLLM:
    def test_usage_tracking(self):
        llm = MockLLM(responses=[])
        llm._track_usage({"prompt_tokens": 100, "completion_tokens": 50})
        llm._track_usage({"prompt_tokens": 200, "completion_tokens": 80})
        assert llm.total_input_tokens == 300
        assert llm.total_output_tokens == 130
        assert llm.usage_summary["total_tokens"] == 430


class TestMockLLM:
    def test_sequential_responses(self):
        llm = MockLLM(responses=[
            LLMResponse(text="First"),
            LLMResponse(text="Second"),
        ])
        r1 = llm.chat([{"role": "user", "content": "hi"}])
        r2 = llm.chat([{"role": "user", "content": "hi again"}])
        assert r1.text == "First"
        assert r2.text == "Second"
        assert len(llm.call_log) == 2
