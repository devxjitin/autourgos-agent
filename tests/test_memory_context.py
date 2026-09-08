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


class _HugeFormatForLlmMemory:
    """Regression fixture for Finding #7: format_for_llm() returns a huge
    blob (bigger than any reasonable scratchpad budget) with no way for the
    memory object itself to bound it -- exercises the agent-side trim."""

    def add_user_message(self, content: str) -> None:
        pass

    def add_agent_message(self, content: str) -> None:
        pass

    def format_for_llm(self, query=None) -> str:
        return "OLDMEMORY" + ("x" * 5000) + "RECENTMEMORY"


def test_huge_memory_context_is_bounded_in_prompt_mode():
    """
    Regression for Finding #7: memory_context used to be baked into the
    rendered prompt with no size limit at all, unlike the scratchpad (which
    _trim_scratchpad already bounded). A memory backend returning a huge
    format_for_llm() blob could blow the context window even with
    max_scratchpad_chars set, since that setting only ever capped the
    scratchpad, not memory.
    """
    memory = _HugeFormatForLlmMemory()
    agent = make_test_agent(
        responses=[_final("hi")], memory=memory, max_scratchpad_chars=500,
    )
    agent.invoke("hello")

    prompt_sent = str(agent.llm.calls[0]["prompt"])
    assert len(prompt_sent) < 5012 + 100  # sanity: didn't just pass the whole blob through
    assert "RECENTMEMORY" in prompt_sent  # tail (most recent) survives the trim
    assert "OLDMEMORY" not in prompt_sent  # head (oldest) is what gets trimmed


def test_huge_memory_context_is_bounded_in_native_mode():
    memory = _HugeFormatForLlmMemory()
    llm = ScriptedToolCallLLM([ScriptedToolCallLLM.final("hi")])
    agent = Agent(llm=llm, tool_calling_mode="native", memory=memory, max_scratchpad_chars=500)
    agent.add_tools({"name": "noop", "description": "no-op", "parameters": {}, "func": lambda: "ok"})
    agent.invoke("hello")

    sent_messages = llm.calls[0]["prompt"]
    combined = json.dumps(sent_messages, default=str)
    assert "RECENTMEMORY" in combined
    assert "OLDMEMORY" not in combined


def test_native_mode_final_call_messages_respect_combined_budget():
    """
    Regression for Finding #7: _trim_native_messages used to size the turns
    budget against the FULL max_scratchpad_chars, then the caller prepended
    system_prompt + memory_context on top afterward, uncounted -- the real
    final call_messages sent to the model could exceed max_scratchpad_chars
    by however large system_prompt/memory_context were. Now the turns
    budget is reduced by the system messages' size first, so the combined
    total (system_messages + trimmed turns) stays within budget.
    """
    memory = _HugeFormatForLlmMemory()
    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("noop", {}),
        ScriptedToolCallLLM.tool_call("noop", {}),
        ScriptedToolCallLLM.tool_call("noop", {}),
        ScriptedToolCallLLM.final("hi"),
    ])
    agent = Agent(
        llm=llm, tool_calling_mode="native", memory=memory,
        max_scratchpad_chars=600, system_prompt="You are a helpful assistant.",
    )
    agent.add_tools({"name": "noop", "description": "no-op", "parameters": {}, "func": lambda: "ok" * 50})
    agent.invoke("hello")

    last_call_messages = llm.calls[-1]["prompt"]
    total_chars = len(json.dumps(last_call_messages, default=str))
    # Some slack for the trim markers themselves (see _trim_text_to_budget's
    # docstring: char cap alone can slightly exceed max_chars by the marker
    # length) -- this asserts it's coordinated, not that it's byte-exact.
    assert total_chars < 600 * 2
