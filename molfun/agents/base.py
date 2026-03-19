"""
Base agent with the tool-calling loop.

The agent connects an LLM (the brain) to MolfunTools (the hands)
via a message-based loop. It handles context management, error
recovery, and experiment persistence.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import json
import time
import logging

from molfun.agents.llm.base import BaseLLM, LLMResponse
from molfun.agents.tools import MolfunTools
from molfun.agents.memory import ExperimentMemory
from molfun.tracking.base import BaseTracker

logger = logging.getLogger(__name__)


class AgentConfig:
    """Configuration for an agent run."""

    def __init__(
        self,
        max_steps: int = 100,
        max_consecutive_errors: int = 3,
        compact_every: int = 15,
        verbose: bool = True,
    ):
        self.max_steps = max_steps
        self.max_consecutive_errors = max_consecutive_errors
        self.compact_every = compact_every
        self.verbose = verbose


class BaseAgent(ABC):
    """
    Agent that uses an LLM to autonomously run experiments.

    The loop:
    1. Send messages + tools to the LLM
    2. If the LLM calls a tool, execute it and append the result
    3. If the LLM sends text, log it as reasoning
    4. Repeat until done, budget exhausted, or max steps
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: MolfunTools,
        memory: Optional[ExperimentMemory] = None,
        config: Optional[AgentConfig] = None,
        tracker: Optional[BaseTracker] = None,
    ):
        self.llm = llm
        self.tools = tools
        self.memory = memory or ExperimentMemory()
        self.config = config or AgentConfig()
        self.tracker = tracker

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt that defines the agent's behavior."""
        ...

    def run(self, objective: str) -> ExperimentMemory:
        """
        Run the agent with a natural language objective.

        Args:
            objective: What the agent should accomplish, e.g.
                "Find the best architecture for binding affinity
                prediction. Try different block types, attention
                mechanisms, and training strategies."

        Returns:
            ExperimentMemory with all experiments and findings.
        """
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": objective},
        ]

        consecutive_errors = 0
        step = 0

        self._log(f"Agent started. Objective: {objective}")
        self._log(f"LLM: {self.llm}")
        self._log(f"Max steps: {self.config.max_steps}")
        self._log("")

        if self.tracker is not None:
            self.tracker.start_run(
                name=f"agent-{type(self).__name__}",
                tags=["agent"],
                config={"objective": objective, "llm": str(self.llm),
                        "max_steps": self.config.max_steps},
            )

        while step < self.config.max_steps:
            step += 1

            if self.tools.is_done:
                self._log(f"\nAgent finished at step {step}.")
                break

            # Inject memory context before each LLM call
            context_msg = self._build_context_message()
            active_messages = messages.copy()
            if context_msg:
                active_messages.insert(1, context_msg)

            try:
                response = self.llm.chat(active_messages, tools=self.tools.schemas)
            except Exception as e:
                consecutive_errors += 1
                self._log(f"  LLM error ({consecutive_errors}): {e}")
                if consecutive_errors >= self.config.max_consecutive_errors:
                    self._log("Max consecutive errors reached. Stopping.")
                    break
                time.sleep(2 ** consecutive_errors)
                continue

            consecutive_errors = 0

            if response.has_tool_calls:
                # Build assistant message with tool calls (for message history)
                assistant_msg = {"role": "assistant", "content": response.text or ""}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages.append(assistant_msg)

                for tc in response.tool_calls:
                    self._log(f"  [{step}] Tool: {tc.name}({_truncate_args(tc.arguments)})")

                    t0 = time.time()
                    result = self.tools.execute(tc.name, tc.arguments)
                    elapsed = time.time() - t0

                    self._log(f"        Result ({elapsed:.1f}s): {_truncate(result, 200)}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                    # Persist experiment if one was just completed
                    exp = self.tools.get_last_experiment()
                    if exp is not None:
                        self.memory.log_experiment(exp)
                        if self.tracker is not None:
                            self.tracker.log_metrics({
                                "experiment_count": self.memory.count,
                                **({"best_val_loss": exp.best_val_loss}
                                   if exp.best_val_loss is not None else {}),
                            }, step=self.memory.count)
                        self.tools.clear_last_experiment()

            elif response.text:
                self._log(f"  [{step}] Agent: {_truncate(response.text, 300)}")
                messages.append({"role": "assistant", "content": response.text})
                self.memory.log_reasoning(response.text)

            # Compact message history periodically
            if step % self.config.compact_every == 0 and step > 0:
                messages = self._compact_messages(messages)

        self._log_final_summary()

        if self.tracker is not None:
            best = self.memory.best()
            if best:
                self.tracker.log_metrics({"final_best_val_loss": best.best_val_loss or 0.0})
            self.tracker.log_text(self.memory.summary_for_context(), tag="final_summary")
            self.tracker.end_run(status="completed")

        return self.memory

    # ------------------------------------------------------------------
    # Context and memory management
    # ------------------------------------------------------------------

    def _build_context_message(self) -> Optional[dict]:
        """Build a system message with current experiment memory."""
        if self.memory.count == 0:
            return None
        summary = self.memory.summary_for_context()
        return {
            "role": "system",
            "content": f"Current experiment state:\n{summary}",
        }

    def _compact_messages(self, messages: list[dict]) -> list[dict]:
        """
        Compact old messages to prevent context window overflow.

        Keeps: system prompt, last N messages, memory summary.
        """
        if len(messages) <= 20:
            return messages

        system = messages[0]
        recent = messages[-15:]

        summary_text = (
            f"[Previous conversation compacted. "
            f"{len(messages) - 16} messages summarized.]\n"
            f"Key context: {self.memory.count} experiments run. "
        )
        best = self.memory.best()
        if best:
            summary_text += f"Best so far: {best.summary_line()}"

        compacted = [
            system,
            {"role": "system", "content": summary_text},
        ] + recent

        self._log(f"  (compacted {len(messages)} → {len(compacted)} messages)")
        return compacted

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg, flush=True)
        logger.info(msg)

    def _log_final_summary(self) -> None:
        self._log("\n" + "=" * 60)
        self._log("AGENT RUN COMPLETE")
        self._log("=" * 60)
        self._log(f"Experiments: {self.memory.count} total, "
                   f"{len(self.memory.completed)} completed, "
                   f"{len(self.memory.failed)} failed")
        best = self.memory.best()
        if best:
            self._log(f"Best: {best.summary_line()}")
            self._log(f"Config: {json.dumps(best.config.to_dict(), indent=2, default=str)}")
        self._log(f"LLM usage: {self.llm.usage_summary}")
        if self.tools.done_summary:
            self._log(f"\nAgent summary:\n{self.tools.done_summary}")
        self._log("=" * 60)


def _truncate(s: str, max_len: int = 200) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _truncate_args(args: dict) -> str:
    s = json.dumps(args, default=str)
    return _truncate(s, 150)
