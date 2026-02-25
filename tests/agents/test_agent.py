"""Tests for the agent loop with mocked LLM and tools."""

import json
import pytest

from molfun.agents.base import BaseAgent, AgentConfig
from molfun.agents.llm.base import BaseLLM, LLMResponse, ToolCall
from molfun.agents.tools import MolfunTools
from molfun.agents.memory import ExperimentMemory
from molfun.agents.researcher import ResearchAgent


class MockLLM(BaseLLM):
    """Scripted LLM that returns pre-defined responses."""

    def __init__(self, responses: list[LLMResponse]):
        super().__init__(model="mock", temperature=0.0)
        self._responses = list(responses)
        self._idx = 0

    def chat(self, messages, tools=None):
        if self._idx < len(self._responses):
            r = self._responses[self._idx]
            self._idx += 1
            return r
        return LLMResponse(tool_calls=[
            ToolCall(id="done_1", name="done",
                     arguments={"summary": "Auto-done after exhausting responses."}),
        ])


class TestAgentLoop:
    def test_agent_calls_tool_and_stops(self):
        """Agent calls list_components, then signals done."""
        llm = MockLLM(responses=[
            LLMResponse(tool_calls=[
                ToolCall(id="tc_1", name="list_components", arguments={}),
            ]),
            LLMResponse(text="I see the available components. Let me try a baseline."),
            LLMResponse(tool_calls=[
                ToolCall(id="tc_2", name="done",
                         arguments={"summary": "Explored components."}),
            ]),
        ])
        tools = MolfunTools(train_loader=None, val_loader=None, device="cpu")
        config = AgentConfig(max_steps=10, verbose=False)
        agent = ResearchAgent(llm=llm, tools=tools, config=config)

        memory = agent.run("Test objective")

        assert tools.is_done
        assert tools.done_summary == "Explored components."
        assert len(memory.reasoning_log) > 0

    def test_agent_respects_max_steps(self):
        """Agent stops at max_steps even if not done."""
        llm = MockLLM(responses=[
            LLMResponse(text=f"Thinking step {i}") for i in range(100)
        ])
        tools = MolfunTools(train_loader=None, val_loader=None, device="cpu")
        config = AgentConfig(max_steps=5, verbose=False)
        agent = ResearchAgent(llm=llm, tools=tools, config=config)

        memory = agent.run("Keep thinking forever")

        assert len(memory.reasoning_log) <= 5

    def test_agent_handles_tool_error(self):
        """Agent continues after a tool returns an error."""
        llm = MockLLM(responses=[
            LLMResponse(tool_calls=[
                ToolCall(id="tc_1", name="save_best_model",
                         arguments={"path": "/tmp/test"}),
            ]),
            LLMResponse(tool_calls=[
                ToolCall(id="tc_2", name="done",
                         arguments={"summary": "Handled the error."}),
            ]),
        ])
        tools = MolfunTools(train_loader=None, val_loader=None, device="cpu")
        config = AgentConfig(max_steps=10, verbose=False)
        agent = ResearchAgent(llm=llm, tools=tools, config=config)

        memory = agent.run("Try saving")
        assert tools.is_done


class TestResearchAgent:
    def test_system_prompt(self):
        llm = MockLLM(responses=[])
        tools = MolfunTools(train_loader=None, val_loader=None, device="cpu")
        agent = ResearchAgent(llm=llm, tools=tools)
        prompt = agent.system_prompt()
        assert "protein" in prompt.lower()
        assert "attention" in prompt.lower()
        assert "tool" in prompt.lower()

    def test_custom_system_prompt(self):
        llm = MockLLM(responses=[])
        tools = MolfunTools(train_loader=None, val_loader=None, device="cpu")
        agent = ResearchAgent(llm=llm, tools=tools, custom_system_prompt="Custom prompt.")
        assert agent.system_prompt() == "Custom prompt."
