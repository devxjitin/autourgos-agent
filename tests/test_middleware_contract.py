"""
Tests for the middleware integration contract added in v1.6.0:
  - self.scratchpad / self.current_query are live, externally-readable
  - on_before_iteration hook: returned kwargs reach the LLM call
  - try/finally ordering fix: on_agent_error / run_end fire even when
    memory.add_user_message() raises
  - make_test_agent() runs a full loop end-to-end, zero network calls
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from autourgos_agent import CallbackHandler, Agent
from autourgos_agent.testing import make_test_agent


# -- scratchpad / current_query are live -------------------------------------

class ScratchpadSpyHandler(CallbackHandler):
    def __init__(self) -> None:
        self.snapshots: List[Dict[str, Any]] = []

    def on_iteration_start(self, iteration, agent=None, **kwargs):
        self.snapshots.append({
            "iteration": iteration,
            "scratchpad": agent.scratchpad,
            "current_query": agent.current_query,
        })


def test_scratchpad_and_query_reflect_loop_state_mid_run():
    responses = [
        json.dumps({
            "thought": "step one",
            "actions": [{"action": "echo", "action_input": {"text": "hi"}}],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "final"}),
    ]
    spy = ScratchpadSpyHandler()
    agent = make_test_agent(responses=responses, middleware=[spy])

    result = agent.invoke("what's up")

    assert result == "final"
    assert len(spy.snapshots) == 2
    # iteration 1 starts with an empty scratchpad
    assert spy.snapshots[0]["scratchpad"] == ""
    assert spy.snapshots[0]["current_query"] == "what's up"
    # iteration 2 sees the scratchpad populated by iteration 1's tool call
    assert "echo" in spy.snapshots[1]["scratchpad"]
    assert spy.snapshots[1]["current_query"] == "what's up"
    # after the run, the final scratchpad is still readable on the agent
    assert "echo" in agent.scratchpad


def test_scratchpad_resets_between_invoke_calls():
    agent = make_test_agent(responses=[
        json.dumps({
            "thought": None,
            "actions": [{"action": "echo", "action_input": {"text": "hi"}}],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "one"}),
    ])
    agent.invoke("first")
    assert "echo" in agent.scratchpad

    agent.llm._responses = [json.dumps({"thought": None, "actions": [], "final_answer": "two"})]
    result = agent.invoke("second")
    assert result == "two"
    # scratchpad must not leak content from the first run
    assert "echo" not in agent.scratchpad
    assert agent.current_query == "second"


# -- on_before_iteration reaches the LLM call ---------------------------------

class InjectingHandler(CallbackHandler):
    def __init__(self, extra: Dict[str, Any]) -> None:
        self._extra = extra
        self.calls = 0

    def on_before_iteration(self, iteration, agent=None, **kwargs) -> Optional[Dict[str, Any]]:
        self.calls += 1
        return dict(self._extra)


def test_on_before_iteration_kwargs_reach_llm_invoke():
    handler = InjectingHandler({"temperature": 0.0, "trace_id": "abc123"})
    agent = make_test_agent(
        responses=[json.dumps({"thought": None, "actions": [], "final_answer": "ok"})],
        middleware=[handler],
    )
    agent.invoke("go")

    assert handler.calls == 1
    assert len(agent.llm.calls) == 1
    received_kwargs = agent.llm.calls[0]["kwargs"]
    assert received_kwargs.get("temperature") == 0.0
    assert received_kwargs.get("trace_id") == "abc123"


def test_on_before_iteration_merge_later_handler_wins():
    h1 = InjectingHandler({"x": 1, "y": 1})
    h2 = InjectingHandler({"y": 2})
    agent = make_test_agent(
        responses=[json.dumps({"thought": None, "actions": [], "final_answer": "ok"})],
        middleware=[h1, h2],
    )
    agent.invoke("go")

    received_kwargs = agent.llm.calls[0]["kwargs"]
    assert received_kwargs["x"] == 1
    assert received_kwargs["y"] == 2  # h2 overrides h1


def test_on_before_iteration_not_persisted_across_iterations():
    class OnceHandler(CallbackHandler):
        def __init__(self):
            self.n = 0

        def on_before_iteration(self, iteration, agent=None, **kwargs):
            self.n += 1
            if self.n == 1:
                return {"only_first": True}
            return None

    handler = OnceHandler()
    agent = make_test_agent(
        responses=[
            json.dumps({
                "thought": None,
                "actions": [{"action": "echo", "action_input": {"text": "x"}}],
                "final_answer": None,
            }),
            json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
        ],
        middleware=[handler],
    )
    agent.invoke("go")

    assert "only_first" in agent.llm.calls[0]["kwargs"]
    assert "only_first" not in agent.llm.calls[1]["kwargs"]


def test_on_before_iteration_none_return_is_noop():
    agent = make_test_agent(
        responses=[json.dumps({"thought": None, "actions": [], "final_answer": "ok"})],
    )
    # default CallbackHandler.on_before_iteration returns None; no middleware attached
    result = agent.invoke("go")
    assert result == "ok"
    assert agent.llm.calls[0]["kwargs"] == {}


# -- try/finally ordering fix --------------------------------------------------

class RaisingMemory:
    def add_user_message(self, message: str) -> None:
        raise RuntimeError("memory boom")

    def add_assistant_message(self, message: str) -> None:
        pass

    def get_history(self):
        return []


def test_error_and_run_end_fire_when_memory_add_user_message_raises():
    events: List[str] = []

    class RecordingHandler(CallbackHandler):
        def on_agent_error(self, error, agent=None, **kwargs):
            events.append("on_agent_error")

    agent = make_test_agent(
        responses=[json.dumps({"thought": None, "actions": [], "final_answer": "ok"})],
        memory=RaisingMemory(),
        middleware=[RecordingHandler()],
    )

    # patch logger.run_end to record it fired
    original_run_end = agent.logger.run_end
    def spy_run_end(*a, **kw):
        events.append("run_end")
        return original_run_end(*a, **kw)
    agent.logger.run_end = spy_run_end

    with pytest.raises(RuntimeError, match="memory boom"):
        agent.invoke("hello")

    assert "on_agent_error" in events
    assert "run_end" in events


@pytest.mark.asyncio
async def test_error_and_run_end_fire_when_memory_add_user_message_raises_async():
    events: List[str] = []

    class RecordingHandler(CallbackHandler):
        def on_agent_error(self, error, agent=None, **kwargs):
            events.append("on_agent_error")

    agent = make_test_agent(
        responses=[json.dumps({"thought": None, "actions": [], "final_answer": "ok"})],
        memory=RaisingMemory(),
        middleware=[RecordingHandler()],
    )

    original_run_end = agent.logger.run_end
    def spy_run_end(*a, **kw):
        events.append("run_end")
        return original_run_end(*a, **kw)
    agent.logger.run_end = spy_run_end

    with pytest.raises(RuntimeError, match="memory boom"):
        await agent.ainvoke("hello")

    assert "on_agent_error" in events
    assert "run_end" in events


# -- make_test_agent end-to-end, zero network calls ----------------------------

def test_make_test_agent_runs_full_loop_to_final_answer():
    agent = make_test_agent(responses=[
        json.dumps({
            "thought": "I'll echo",
            "actions": [{"action": "echo", "action_input": {"text": "hello"}}],
            "final_answer": None,
        }),
        json.dumps({"thought": "done", "actions": [], "final_answer": "the final answer"}),
    ])

    result = agent.invoke("do the thing")

    assert result == "the final answer"
    assert isinstance(agent, Agent)
    assert agent.llm.call_count == 2


@pytest.mark.asyncio
async def test_make_test_agent_async_runs_full_loop():
    agent = make_test_agent(responses=[
        json.dumps({"thought": None, "actions": [], "final_answer": "async done"}),
    ])
    result = await agent.ainvoke("go async")
    assert result == "async done"


class _AgentMessageOnlyMemory:
    """Mimics the autourgos-memory family: add_agent_message, no add_assistant_message."""

    def __init__(self):
        self.messages = []

    def add_user_message(self, message):
        self.messages.append(("user", message))

    def add_agent_message(self, message):
        self.messages.append(("agent", message))

    def get_history(self):
        return self.messages


def test_memory_family_add_agent_message_is_used():
    from autourgos_agent.testing import make_test_agent

    memory = _AgentMessageOnlyMemory()
    agent = make_test_agent(
        responses=['{"thought": "done", "actions": [], "final_answer": "hi there"}'],
        memory=memory,
    )
    result = agent.invoke("hello")
    assert result == "hi there"
    assert ("agent", "hi there") in memory.messages
