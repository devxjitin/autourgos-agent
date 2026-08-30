"""
Tests that `memory=` is actually read back into the prompt/messages sent to
the LLM, not just written to. Covers both memory conventions the ecosystem
uses: this package's own MemoryProtocol (get_history() -> list of dicts or
(role, content) tuples) and the autourgos-memory family's BaseMemory
(format_for_llm()/get_context()).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from autourgos_agent import Agent
from autourgos_agent.testing import make_test_agent, ScriptedToolCallLLM


class DictHistoryMemory:
    """MemoryProtocol-shaped memory, per the README's own example."""

    def __init__(self) -> None:
        self._history: List[Dict[str, str]] = []

    def add_user_message(self, message: str) -> None:
        self._history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str) -> None:
        self._history.append({"role": "assistant", "content": message})

    def get_history(self) -> List[Dict[str, str]]:
        return list(self._history)


class TupleHistoryMemory:
    """MemoryProtocol-shaped memory whose get_history() returns tuples."""

    def __init__(self) -> None:
        self.messages: List[Any] = []

    def add_user_message(self, message: str) -> None:
        self.messages.append(("user", message))

    def add_agent_message(self, message: str) -> None:
        self.messages.append(("agent", message))

    def get_history(self) -> List[Any]:
        return self.messages


class FormatForLlmMemory:
    """autourgos-memory family (BaseMemory) shaped memory."""

    def __init__(self) -> None:
        self._lines: List[str] = []

    def add_user_message(self, content: str) -> None:
        self._lines.append(f"user: {content}")

    def add_agent_message(self, content: str) -> None:
        self._lines.append(f"agent: {content}")

    def add_tool_message(self, tool_name: str, result: str) -> None:
        self._lines.append(f"tool[{tool_name}]: {result}")

    def clear(self) -> None:
        self._lines = []

    def format_for_llm(self, query=None) -> str:
        if not self._lines:
            return ""
        return "\n--- Previous Conversation Context ---\n" + "\n".join(self._lines) + "\n"


def _final(text: str) -> str:
    return json.dumps({"thought": None, "actions": [], "final_answer": text})


def test_dict_history_memory_reaches_prompt_mode_prompt():
    memory = DictHistoryMemory()
    agent = make_test_agent(responses=[_final("The capital of France is Paris.")], memory=memory)
    agent.invoke("Search for the capital of France.")

    agent2 = make_test_agent(responses=[_final("???")], memory=memory)
    agent2.invoke("What city did I just ask about?")

    prompt_sent = str(agent2.llm.calls[0]["prompt"])
    assert "Paris" in prompt_sent


def test_tuple_history_memory_reaches_prompt_mode_prompt():
    memory = TupleHistoryMemory()
    agent = make_test_agent(responses=[_final("The capital of France is Paris.")], memory=memory)
    agent.invoke("Search for the capital of France.")

    agent2 = make_test_agent(responses=[_final("???")], memory=memory)
    agent2.invoke("What city did I just ask about?")

    prompt_sent = str(agent2.llm.calls[0]["prompt"])
    assert "Paris" in prompt_sent


def test_format_for_llm_memory_reaches_prompt_mode_prompt():
    memory = FormatForLlmMemory()
    agent = make_test_agent(responses=[_final("The capital of France is Paris.")], memory=memory)
    agent.invoke("Search for the capital of France.")

    agent2 = make_test_agent(responses=[_final("???")], memory=memory)
    agent2.invoke("What city did I just ask about?")

    prompt_sent = str(agent2.llm.calls[0]["prompt"])
    assert "Paris" in prompt_sent


def test_no_memory_context_block_when_no_memory_attached():
    agent = make_test_agent(responses=[_final("hi")], memory=None)
    agent.invoke("hello")

    prompt_sent = str(agent.llm.calls[0]["prompt"])
    assert "Previous Conversation Context" not in prompt_sent


def test_memory_context_reaches_native_mode_messages():
    memory = DictHistoryMemory()
    llm = ScriptedToolCallLLM([ScriptedToolCallLLM.final("The capital of France is Paris.")])
    agent = Agent(llm=llm, tool_calling_mode="native", memory=memory)
    agent.add_tools({"name": "noop", "description": "no-op", "parameters": {}, "func": lambda: "ok"})
    agent.invoke("Search for the capital of France.")

    llm2 = ScriptedToolCallLLM([ScriptedToolCallLLM.final("???")])
    agent2 = Agent(llm=llm2, tool_calling_mode="native", memory=memory)
    agent2.add_tools({"name": "noop", "description": "no-op", "parameters": {}, "func": lambda: "ok"})
    agent2.invoke("What city did I just ask about?")

    sent_messages = llm2.calls[0]["prompt"]
    assert any("Paris" in str(m.get("content", "")) for m in sent_messages)
