"""
Tests for prompt-mode (tool_calling_mode="prompt", the default) tool
concurrency and the exception-based stop-condition contract.

Regression coverage:
  - prompt.py tells the model "You can call multiple tools at once if they
    don't depend on each other's outputs", but the prompt-mode loop used to
    run actions in a plain sequential `for` loop -- 3 independent 0.1s tools
    took ~0.3s instead of ~0.1s. It's now run through a ThreadPoolExecutor
    (sync loop) / asyncio.gather (async loop), mirroring how native mode
    already worked, so the engine matches what the prompt promises.
  - Timeout/MaxIterations/ParseError/LLMError used to be signaled by
    returning a "[Tag] message" string from invoke()/ainvoke(), which
    callers had to string-sniff. They're now raised as AgentTimeoutError /
    AgentMaxIterationsError / AgentParseError / AgentLLMError.
  - approval_callback can now be an async function in the async loop
    (ainvoke()); a sync one still works unchanged in both invoke()/ainvoke().
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from autourgos_agent import (
    Agent,
    AgentLLMError,
    AgentMaxIterationsError,
    AgentParseError,
    AgentTimeoutError,
)
from autourgos_agent.testing import ScriptedFakeLLM


def _tool(name: str, func, description: str = "d"):
    return {"name": name, "description": description, "parameters": {}, "func": func}


def _slow(n: int) -> int:
    time.sleep(0.1)
    return n


def _multi_tool_call_responses(n: int):
    return [
        json.dumps({
            "thought": "run independent tools",
            "actions": [{"action": f"t{i}", "action_input": {"n": i}} for i in range(n)],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
    ]


# -- concurrent multi-tool-call execution (prompt mode) -----------------------

def test_prompt_loop_executes_multiple_tool_calls_concurrently():
    tools = [_tool(f"t{i}", _slow) for i in range(3)]
    agent = Agent(llm=ScriptedFakeLLM(_multi_tool_call_responses(3)), max_iterations=5)
    agent.add_tools(*tools)

    start = time.monotonic()
    result = agent.invoke("run 3 independent tools")
    elapsed = time.monotonic() - start

    assert result == "done"
    assert elapsed < 0.25  # ~0.1s if truly concurrent, ~0.3s if sequential


async def _slow_async(n: int) -> int:
    await asyncio.sleep(0.1)
    return n


@pytest.mark.asyncio
async def test_prompt_loop_async_executes_multiple_tool_calls_concurrently():
    tools = [_tool(f"t{i}", _slow_async) for i in range(3)]
    agent = Agent(llm=ScriptedFakeLLM(_multi_tool_call_responses(3)), max_iterations=5)
    agent.add_tools(*tools)

    start = time.monotonic()
    result = await agent.ainvoke("run 3 independent tools")
    elapsed = time.monotonic() - start

    assert result == "done"
    assert elapsed < 0.25


# -- exception-based stop conditions -------------------------------------------

def test_prompt_loop_max_iterations_raises():
    responses = [
        json.dumps({"thought": "still thinking", "actions": [], "final_answer": None})
        for _ in range(10)
    ]
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=2)
    agent.add_tools(_tool("noop", lambda: None))

    with pytest.raises(AgentMaxIterationsError):
        agent.invoke("never finish")


def test_prompt_loop_timeout_raises():
    class SlowLLM:
        """Keeps calling the 'noop' tool (never a final answer) so the loop
        runs several iterations -- the timeout guard only runs at the top
        of each iteration, so a single slow call that returns immediately
        wouldn't get a second chance to trip it."""

        def invoke(self, prompt, **kwargs):
            time.sleep(0.05)
            return json.dumps({
                "thought": None,
                "actions": [{"action": "noop", "action_input": {}}],
                "final_answer": None,
            })

        async def ainvoke(self, prompt, **kwargs):
            return self.invoke(prompt, **kwargs)

    agent = Agent(llm=SlowLLM(), max_iterations=20, max_execution_time=0.1)
    agent.add_tools(_tool("noop", lambda: None))

    with pytest.raises(AgentTimeoutError):
        agent.invoke("go")


def test_prompt_loop_parse_error_raises():
    agent = Agent(llm=ScriptedFakeLLM(["not json"] * 5), max_iterations=10, max_consecutive_parse_errors=2)
    agent.add_tools(_tool("noop", lambda: None))

    with pytest.raises(AgentParseError):
        agent.invoke("confuse me")


def test_prompt_loop_llm_error_raises():
    class RaisingLLM:
        def invoke(self, prompt, **kwargs):
            raise RuntimeError("boom")

        async def ainvoke(self, prompt, **kwargs):
            raise RuntimeError("boom")

    agent = Agent(llm=RaisingLLM(), max_iterations=5)
    agent.add_tools(_tool("noop", lambda: None))

    with pytest.raises(AgentLLMError):
        agent.invoke("go")


# -- malformed `actions` shape --------------------------------------------------

def test_prompt_loop_malformed_actions_shape_is_treated_as_parse_error():
    """A response with `actions` as a single object (not a list of dicts) used
    to pass the `if not actions` truthiness check, then crash with
    AttributeError when the loop did `for action_dict in actions:
    action_dict.get(...)` over the dict's string keys. It must instead be
    treated the same as any other malformed response."""
    responses = [
        json.dumps({
            "thought": "oops",
            "actions": {"action": "noop", "action_input": {}},
            "final_answer": None,
        })
    ] * 5
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=10, max_consecutive_parse_errors=2)
    agent.add_tools(_tool("noop", lambda: None))

    with pytest.raises(AgentParseError):
        agent.invoke("confuse me with a dict instead of a list")


def test_prompt_loop_actions_with_non_dict_items_is_treated_as_parse_error():
    responses = [
        json.dumps({"thought": None, "actions": ["noop"], "final_answer": None})
    ] * 5
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=10, max_consecutive_parse_errors=2)
    agent.add_tools(_tool("noop", lambda: None))

    with pytest.raises(AgentParseError):
        agent.invoke("confuse me with a list of strings")


# -- explicit max_iterations=0 --------------------------------------------------

def test_invoke_explicit_max_iterations_zero_is_respected():
    """max_iterations=0 is falsy, so `max_iterations or self.max_iterations`
    used to silently fall back to the instance default instead of honoring
    the explicit override."""
    responses = [
        json.dumps({"thought": "still thinking", "actions": [], "final_answer": None})
        for _ in range(10)
    ]
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=5)
    agent.add_tools(_tool("noop", lambda: None))

    with pytest.raises(AgentMaxIterationsError):
        agent.invoke("never finish", max_iterations=0)


@pytest.mark.asyncio
async def test_ainvoke_explicit_max_iterations_zero_is_respected():
    responses = [
        json.dumps({"thought": "still thinking", "actions": [], "final_answer": None})
        for _ in range(10)
    ]
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=5)
    agent.add_tools(_tool("noop", lambda: None))

    with pytest.raises(AgentMaxIterationsError):
        await agent.ainvoke("never finish", max_iterations=0)


@pytest.mark.asyncio
async def test_async_prompt_loop_llm_error_raises():
    class RaisingLLM:
        def invoke(self, prompt, **kwargs):
            raise RuntimeError("boom")

        async def ainvoke(self, prompt, **kwargs):
            raise RuntimeError("boom")

    agent = Agent(llm=RaisingLLM(), max_iterations=5)
    agent.add_tools(_tool("noop", lambda: None))

    with pytest.raises(AgentLLMError):
        await agent.ainvoke("go")


# -- async approval_callback ----------------------------------------------------

@pytest.mark.asyncio
async def test_async_loop_awaits_async_approval_callback():
    calls = []

    async def approve(name, tool_input):
        await asyncio.sleep(0.01)
        calls.append((name, tool_input))
        return True

    responses = [
        json.dumps({
            "thought": "call it",
            "actions": [{"action": "echo", "action_input": {"text": "hi"}}],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
    ]
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=5, approval_callback=approve)
    agent.add_tools(_tool("echo", lambda text: text))

    result = await agent.ainvoke("do it")

    assert result == "done"
    assert calls == [("echo", {"text": "hi"})]
    assert "Observation: Tool call was denied" not in agent.scratchpad


@pytest.mark.asyncio
async def test_async_loop_async_approval_callback_can_deny():
    async def deny(name, tool_input):
        return False

    responses = [
        json.dumps({
            "thought": "call it",
            "actions": [{"action": "echo", "action_input": {"text": "hi"}}],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
    ]
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=5, approval_callback=deny)
    agent.add_tools(_tool("echo", lambda text: text))

    result = await agent.ainvoke("do it")

    assert result == "done"
    assert "Observation: Tool call was denied by the approval callback." in agent.scratchpad


@pytest.mark.asyncio
async def test_async_loop_sync_approval_callback_still_works():
    calls = []

    def approve(name, tool_input):
        calls.append((name, tool_input))
        return True

    responses = [
        json.dumps({
            "thought": "call it",
            "actions": [{"action": "echo", "action_input": {"text": "hi"}}],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
    ]
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=5, approval_callback=approve)
    agent.add_tools(_tool("echo", lambda text: text))

    result = await agent.ainvoke("do it")

    assert result == "done"
    assert calls == [("echo", {"text": "hi"})]
