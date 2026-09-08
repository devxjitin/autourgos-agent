"""
Sprint 4b (FRAMEWORK_REVIEW.md Finding #5) regression coverage.

A CallbackHandler.on_iteration implementation can now return a truthy value
to tell the loop "discard this turn's action batch, don't dispatch it" --
e.g. because the handler just injected a newer human instruction and the
actions the LLM already planned were reasoned out before that instruction
arrived. Covers:
  - the discarded turn's tools are never executed, a scratchpad note is
    left instead, and the run recovers normally on the next iteration;
  - a handler returning None/falsy (the overwhelming existing case, and
    every current in-repo middleware) is completely unaffected -- actions
    dispatch exactly as before;
  - on_iteration now fires every iteration, even when the LLM response had
    no thought, so the discard signal can't be missed on a thought-less
    turn (previously this hook was gated behind `if thought:`);
  - parity between the sync and async loops.
"""
from __future__ import annotations

import json

import pytest

from autourgos_agent import Agent, CallbackHandler
from autourgos_agent.testing import ScriptedFakeLLM


def _tool(name: str, func, description: str = "d"):
    return {"name": name, "description": description, "parameters": {}, "func": func}


def _action_then_done_responses():
    return [
        json.dumps({
            "thought": "call the tool",
            "actions": [{"action": "spy", "action_input": {}}],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
    ]


class DiscardOnce(CallbackHandler):
    """Discards exactly the first iteration's action batch, no-ops after."""

    def __init__(self) -> None:
        self.calls = []
        self._fired = False

    def on_iteration(self, iteration, thought, agent=None, **kwargs):
        self.calls.append(iteration)
        if not self._fired:
            self._fired = True
            return True
        return False


class NeverDiscard(CallbackHandler):
    def __init__(self) -> None:
        self.calls = []

    def on_iteration(self, iteration, thought, agent=None, **kwargs):
        self.calls.append(iteration)
        return None


def test_sync_loop_discards_stale_batch_and_recovers():
    tool_calls = []
    tools = [_tool("spy", lambda: tool_calls.append(1) or "ok")]
    handler = DiscardOnce()
    responses = _action_then_done_responses() + [
        json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
    ]
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=5, middleware=[handler])
    agent.add_tools(*tools)

    result = agent.invoke("go")

    assert result == "done"
    assert tool_calls == []  # the discarded batch's tool never ran
    assert "discarded before execution" in agent.scratchpad
    # on_iteration fired at least twice (discard iteration + the recovery one)
    assert len(handler.calls) >= 2


@pytest.mark.asyncio
async def test_async_loop_discards_stale_batch_and_recovers():
    tool_calls = []

    async def spy_tool():
        tool_calls.append(1)
        return "ok"

    tools = [_tool("spy", spy_tool)]
    handler = DiscardOnce()
    responses = _action_then_done_responses() + [
        json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
    ]
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=5, middleware=[handler])
    agent.add_tools(*tools)

    result = await agent.ainvoke("go")

    assert result == "done"
    assert tool_calls == []
    assert "discarded before execution" in agent.scratchpad


def test_handler_returning_none_does_not_discard_sync():
    """Regression guard: a handler that never signals discard (every
    existing in-repo middleware) must leave dispatch completely unaffected."""
    tool_calls = []
    tools = [_tool("spy", lambda: tool_calls.append(1) or "ok")]
    handler = NeverDiscard()
    agent = Agent(llm=ScriptedFakeLLM(_action_then_done_responses()), max_iterations=5, middleware=[handler])
    agent.add_tools(*tools)

    result = agent.invoke("go")

    assert result == "done"
    assert tool_calls == [1]
    assert "discarded before execution" not in agent.scratchpad


@pytest.mark.asyncio
async def test_handler_returning_none_does_not_discard_async():
    tool_calls = []

    async def spy_tool():
        tool_calls.append(1)
        return "ok"

    tools = [_tool("spy", spy_tool)]
    handler = NeverDiscard()
    agent = Agent(llm=ScriptedFakeLLM(_action_then_done_responses()), max_iterations=5, middleware=[handler])
    agent.add_tools(*tools)

    result = await agent.ainvoke("go")

    assert result == "done"
    assert tool_calls == [1]


def test_on_iteration_fires_even_without_a_thought():
    """Previously on_iteration only fired when the response also produced a
    thought -- gating out the discard opportunity on a thought-less turn.
    It must now fire every iteration regardless."""
    handler = NeverDiscard()
    responses = [
        json.dumps({"thought": None, "actions": [{"action": "noop", "action_input": {}}], "final_answer": None}),
        json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
    ]
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=5, middleware=[handler])
    agent.add_tools(_tool("noop", lambda: "ok"))

    result = agent.invoke("go")

    assert result == "done"
    assert handler.calls == [1, 2]


def test_discard_does_not_count_as_a_parse_error():
    """A discarded batch must not increment consecutive_parse_errors --
    repeated legitimate discards (e.g. several late instructions in a row)
    must never trip AgentParseError/AgentMaxIterationsError just for that."""

    class AlwaysDiscard(CallbackHandler):
        def on_iteration(self, iteration, thought, agent=None, **kwargs):
            return True

    responses = [
        json.dumps({
            "thought": "plan",
            "actions": [{"action": "spy", "action_input": {}}],
            "final_answer": None,
        })
        for _ in range(3)
    ] + [json.dumps({"thought": None, "actions": [], "final_answer": "done"})]

    agent = Agent(
        llm=ScriptedFakeLLM(responses),
        max_iterations=10,
        max_consecutive_parse_errors=2,
        middleware=[AlwaysDiscard()],
    )
    agent.add_tools(_tool("spy", lambda: "ok"))

    result = agent.invoke("go")

    assert result == "done"
