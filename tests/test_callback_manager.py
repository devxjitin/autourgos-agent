"""
Tests for the CallbackManager / CallbackHandler lifecycle in
autourgos-agent.

Covers:
  (a) all 9/10 hooks fire during a normal run, with `agent=` passed correctly
  (b) a handler that raises inside a hook does not crash the agent loop
  (c) an old-style handler (narrower signature, no `agent` kwarg) still works
  (d) on_agent_error / on_tool_error fire on failure paths
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from autourgos_agent import Agent, CallbackHandler


# -- test doubles -------------------------------------------------------------

class FakeLLM:
    """Returns a scripted sequence of responses, one per invoke() call."""

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, prompt: Any, **kwargs: Any) -> str:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return json.dumps({"thought": None, "actions": [], "final_answer": "done"})

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> str:
        return self.invoke(prompt, **kwargs)


class RaisingLLM:
    """Always raises on invoke()."""

    def invoke(self, prompt: Any, **kwargs: Any) -> str:
        raise RuntimeError("boom-llm")

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> str:
        raise RuntimeError("boom-llm")


def echo_tool(text: str) -> str:
    return f"echo: {text}"


def failing_tool(text: str) -> str:
    raise ValueError("boom-tool")


ECHO_TOOL_DICT: Dict[str, Any] = {
    "name": "echo",
    "description": "Echo the given text.",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    "func": echo_tool,
}

FAILING_TOOL_DICT: Dict[str, Any] = {
    "name": "failing_tool",
    "description": "A tool that always raises.",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    "func": failing_tool,
}


class RecordingHandler(CallbackHandler):
    """Captures every hook call and the kwargs it received."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def _record(self, name: str, agent: Any, kwargs: Dict[str, Any]) -> None:
        self.calls.append({"hook": name, "agent": agent, "kwargs": kwargs})

    def on_agent_start(self, query, agent=None, **kwargs):
        self._record("on_agent_start", agent, kwargs)

    def on_agent_end(self, result, agent=None, **kwargs):
        self._record("on_agent_end", agent, kwargs)

    def on_agent_error(self, error, agent=None, **kwargs):
        self._record("on_agent_error", agent, kwargs)

    def on_tool_start(self, tool_name, tool_input, agent=None, **kwargs):
        self._record("on_tool_start", agent, kwargs)

    def on_tool_end(self, tool_name, result, agent=None, **kwargs):
        self._record("on_tool_end", agent, kwargs)

    def on_tool_error(self, tool_name, error, agent=None, **kwargs):
        self._record("on_tool_error", agent, kwargs)

    def on_iteration_start(self, iteration, agent=None, **kwargs):
        self._record("on_iteration_start", agent, kwargs)

    def on_iteration(self, iteration, thought, agent=None, **kwargs):
        self._record("on_iteration", agent, kwargs)

    def on_llm_end(self, response, agent=None, **kwargs):
        self._record("on_llm_end", agent, kwargs)

    def on_parse_error(self, iteration, raw_response, agent=None, **kwargs):
        self._record("on_parse_error", agent, kwargs)

    def hook_names(self) -> List[str]:
        return [c["hook"] for c in self.calls]


class ExplodingHandler(CallbackHandler):
    """Raises inside every hook it implements."""

    def on_agent_start(self, query, agent=None, **kwargs):
        raise RuntimeError("handler blew up on_agent_start")

    def on_iteration_start(self, iteration, agent=None, **kwargs):
        raise RuntimeError("handler blew up on_iteration_start")

    def on_tool_start(self, tool_name, tool_input, agent=None, **kwargs):
        raise RuntimeError("handler blew up on_tool_start")

    def on_agent_end(self, result, agent=None, **kwargs):
        raise RuntimeError("handler blew up on_agent_end")


class OldStyleHandler(CallbackHandler):
    """
    Mimics a handler written against the ORIGINAL 6-hook interface,
    before `agent=` support existed -- narrower signatures that do NOT
    accept an `agent` kwarg at all.
    """

    def __init__(self) -> None:
        self.saw: List[str] = []

    def on_agent_start(self, query: str, **kwargs) -> None:
        self.saw.append("on_agent_start")

    def on_agent_end(self, result: str, **kwargs) -> None:
        self.saw.append("on_agent_end")

    def on_tool_start(self, tool_name, tool_input, **kwargs) -> None:
        self.saw.append("on_tool_start")

    def on_tool_end(self, tool_name, result, **kwargs) -> None:
        self.saw.append("on_tool_end")

    def on_iteration(self, iteration, thought, **kwargs) -> None:
        self.saw.append("on_iteration")

    def on_parse_error(self, iteration, raw_response, **kwargs) -> None:
        self.saw.append("on_parse_error")


def make_agent(llm, tools, handlers) -> Agent:
    agent = Agent(llm=llm, middleware=handlers, max_iterations=5)
    agent.add_tools(*tools)
    return agent


# -- (a) full lifecycle, agent= passed correctly ------------------------------

def test_all_hooks_fire_on_successful_run():
    responses = [
        json.dumps({
            "thought": "I should echo",
            "actions": [{"action": "echo", "action_input": {"text": "hi"}}],
            "final_answer": None,
        }),
        json.dumps({"thought": "done thinking", "actions": [], "final_answer": "final result"}),
    ]
    llm = FakeLLM(responses)
    rec = RecordingHandler()
    agent = make_agent(llm, [ECHO_TOOL_DICT], [rec])

    result = agent.invoke("do something")

    assert result == "final result"

    fired = set(rec.hook_names())
    expected = {
        "on_agent_start", "on_agent_end", "on_tool_start", "on_tool_end",
        "on_iteration_start", "on_iteration", "on_llm_end",
    }
    assert expected.issubset(fired), f"missing hooks: {expected - fired}"

    # agent= must be the Agent instance for every recorded call
    for call in rec.calls:
        assert call["agent"] is agent, f"{call['hook']} did not receive agent="

    # on_llm_end must carry the raw LLM response through kwargs so
    # cost/usage middleware doesn't need to reach into agent.llm internals
    llm_end_calls = [c for c in rec.calls if c["hook"] == "on_llm_end"]
    assert llm_end_calls, "on_llm_end never fired"
    assert all("raw" in c["kwargs"] for c in llm_end_calls)


def test_on_llm_end_receives_usage_metadata_from_dict_response():
    """When the LLM wrapper returns the autourgos-openaichat/-responses dict
    shape (response/input_tokens/output_tokens/total_cost/latency_ms), those
    fields must reach on_llm_end as kwargs, not just the extracted text."""

    class DictFakeLLM:
        def invoke(self, prompt, **kwargs):
            return {
                "response": json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
                "provider_used": "openai",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_cost": 0.0002,
                "latency_ms": 123.4,
            }

        async def ainvoke(self, prompt, **kwargs):
            return self.invoke(prompt, **kwargs)

    rec = RecordingHandler()
    agent = make_agent(DictFakeLLM(), [ECHO_TOOL_DICT], [rec])
    agent.invoke("do something")

    llm_end_calls = [c for c in rec.calls if c["hook"] == "on_llm_end"]
    assert llm_end_calls
    kwargs = llm_end_calls[-1]["kwargs"]
    assert kwargs["input_tokens"] == 10
    assert kwargs["output_tokens"] == 5
    assert kwargs["total_cost"] == 0.0002
    assert kwargs["latency_ms"] == 123.4
    assert kwargs["provider_used"] == "openai"


# -- (b) handler exceptions don't crash the loop ------------------------------

def test_exploding_handler_does_not_crash_loop():
    responses = [
        json.dumps({"thought": None, "actions": [], "final_answer": "ok despite explosion"}),
    ]
    llm = FakeLLM(responses)
    agent = make_agent(llm, [ECHO_TOOL_DICT], [ExplodingHandler()])

    result = agent.invoke("hello")
    assert result == "ok despite explosion"


# -- (c) old-style (no `agent` kwarg) handler still works ---------------------

def test_old_style_handler_backward_compatible():
    responses = [
        json.dumps({
            "thought": "using tool",
            "actions": [{"action": "echo", "action_input": {"text": "hi"}}],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "wrapped up"}),
    ]
    llm = FakeLLM(responses)
    old_handler = OldStyleHandler()
    agent = make_agent(llm, [ECHO_TOOL_DICT], [old_handler])

    result = agent.invoke("go")

    assert result == "wrapped up"
    assert "on_agent_start" in old_handler.saw
    assert "on_agent_end" in old_handler.saw
    assert "on_tool_start" in old_handler.saw
    assert "on_tool_end" in old_handler.saw


# -- (d) on_agent_error / on_tool_error fire on failure paths -----------------

def test_on_agent_error_fires_when_llm_raises():
    from autourgos_agent import AgentLLMError

    rec = RecordingHandler()
    agent = make_agent(RaisingLLM(), [ECHO_TOOL_DICT], [rec])

    with pytest.raises(AgentLLMError):
        agent.invoke("trigger llm failure")

    assert "on_agent_error" in rec.hook_names()
    error_calls = [c for c in rec.calls if c["hook"] == "on_agent_error"]
    assert error_calls[0]["agent"] is agent


def test_on_tool_error_fires_when_tool_raises():
    responses = [
        json.dumps({
            "thought": "calling failing tool",
            "actions": [{"action": "failing_tool", "action_input": {"text": "x"}}],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "recovered"}),
    ]
    llm = FakeLLM(responses)
    rec = RecordingHandler()
    agent = make_agent(llm, [FAILING_TOOL_DICT], [rec])

    result = agent.invoke("trigger tool failure")

    assert result == "recovered"
    assert "on_tool_error" in rec.hook_names()
    error_calls = [c for c in rec.calls if c["hook"] == "on_tool_error"]
    assert error_calls[0]["agent"] is agent


# -- async variants ------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_hooks_fire_on_successful_async_run():
    responses = [
        json.dumps({"thought": "thinking", "actions": [], "final_answer": "async result"}),
    ]
    llm = FakeLLM(responses)
    rec = RecordingHandler()
    agent = make_agent(llm, [ECHO_TOOL_DICT], [rec])

    result = await agent.ainvoke("do it async")

    assert result == "async result"
    fired = set(rec.hook_names())
    assert {"on_agent_start", "on_agent_end", "on_iteration_start", "on_llm_end"}.issubset(fired)
