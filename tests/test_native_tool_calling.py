"""
Tests for Agent(tool_calling_mode="native") -- Phase 2 of the
native-tool-calling roadmap. Uses ScriptedToolCallLLM for fast, no-network
unit tests, plus real OpenAIChatModel/OpenAIResponse (mocked client) for
end-to-end integration, mirroring test_openaichat_integration.py's style.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

import pytest

from autourgos_agent import CallbackHandler, Agent
from autourgos_agent.testing import ScriptedToolCallLLM


def _tool(name: str, func, description: str = "d"):
    return {"name": name, "description": description, "parameters": {}, "func": func}


ADD_TOOL = _tool("add", lambda a, b: a + b)


# -- construction ---------------------------------------------------------------

def test_invalid_tool_calling_mode_raises():
    with pytest.raises(ValueError):
        Agent(llm=None, tool_calling_mode="bogus")


def test_default_mode_is_prompt():
    agent = Agent(llm=None)
    assert agent.tool_calling_mode == "prompt"


# -- basic native loop ------------------------------------------------------------

def test_native_loop_tool_call_then_final_answer():
    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("add", {"a": 2, "b": 3}, call_id="c1"),
        ScriptedToolCallLLM.final("5"),
    ])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5)
    agent.add_tools(ADD_TOOL)
    assert agent.invoke("what is 2+3?") == "5"
    assert "add" in agent.scratchpad
    assert "5" in agent.scratchpad


def test_native_loop_no_tools_needed_answers_directly():
    llm = ScriptedToolCallLLM([ScriptedToolCallLLM.final("just an answer")])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5)
    agent.add_tools(ADD_TOOL)
    assert agent.invoke("no tools needed") == "just an answer"


@pytest.mark.asyncio
async def test_native_loop_async():
    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("add", {"a": 10, "b": 20}, call_id="c1"),
        ScriptedToolCallLLM.final("30"),
    ])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5)
    agent.add_tools(ADD_TOOL)
    assert await agent.ainvoke("10+20?") == "30"


# -- concurrent multi-tool-call execution -------------------------------------------

def _slow(n: int) -> int:
    time.sleep(0.1)
    return n


def test_native_loop_executes_multiple_tool_calls_concurrently():
    tools = [_tool(f"t{i}", _slow) for i in range(3)]
    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.calls_([{"name": f"t{i}", "arguments": {"n": i}, "call_id": f"c{i}"} for i in range(3)]),
        ScriptedToolCallLLM.final("done"),
    ])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5)
    agent.add_tools(*tools)

    start = time.monotonic()
    result = agent.invoke("run 3 independent tools")
    elapsed = time.monotonic() - start

    assert result == "done"
    assert elapsed < 0.25  # ~0.1s if truly concurrent, ~0.3s if sequential


async def _slow_async(n: int) -> int:
    # A sync blocking tool (time.sleep) run via asyncio.gather does NOT
    # truly parallelize -- it blocks the event loop just like the
    # pre-existing prompt-mode async loop's _execute_tool_async does.
    # Real concurrency in the async path needs an async tool function
    # (awaited directly) or -- for a sync one -- running it in an executor,
    # which _execute_tool_async doesn't do today. Use an async tool here to
    # test what asyncio.gather in this loop actually parallelizes.
    await asyncio.sleep(0.1)
    return n


@pytest.mark.asyncio
async def test_native_loop_async_executes_multiple_tool_calls_concurrently():
    tools = [_tool(f"t{i}", _slow_async) for i in range(3)]
    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.calls_([{"name": f"t{i}", "arguments": {"n": i}, "call_id": f"c{i}"} for i in range(3)]),
        ScriptedToolCallLLM.final("done"),
    ])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5)
    agent.add_tools(*tools)

    start = time.monotonic()
    result = await agent.ainvoke("run 3 independent tools")
    elapsed = time.monotonic() - start

    assert result == "done"
    assert elapsed < 0.25


# -- approval callback --------------------------------------------------------------

def test_native_loop_approval_callback_deny():
    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("add", {"a": 1, "b": 1}, call_id="c1"),
        ScriptedToolCallLLM.final("denied-path"),
    ])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5, approval_callback=lambda name, inp: False)
    agent.add_tools(ADD_TOOL)
    assert agent.invoke("add stuff") == "denied-path"
    assert "denied" in agent.scratchpad.lower()


# -- memory -------------------------------------------------------------------------

def test_native_loop_records_to_memory():
    class FakeMemory:
        def __init__(self):
            self.msgs = []
        def add_user_message(self, m): self.msgs.append(("user", m))
        def add_agent_message(self, m): self.msgs.append(("agent", m))
        def get_history(self): return self.msgs

    memory = FakeMemory()
    llm = ScriptedToolCallLLM([ScriptedToolCallLLM.final("ok")])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5, memory=memory)
    agent.add_tools(ADD_TOOL)
    agent.invoke("hi")
    assert memory.msgs == [("user", "hi"), ("agent", "ok")]


# -- exhaustion / error paths ---------------------------------------------------------

def test_native_loop_max_iterations_exhaustion():
    from autourgos_agent import AgentMaxIterationsError

    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("add", {"a": 1, "b": 1}, call_id=f"c{i}") for i in range(5)
    ])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=2)
    agent.add_tools(ADD_TOOL)
    with pytest.raises(AgentMaxIterationsError):
        agent.invoke("loop forever")


def test_native_loop_empty_response_exhaustion():
    from autourgos_agent import AgentEmptyResponseError

    llm = ScriptedToolCallLLM([ScriptedToolCallLLM.empty() for _ in range(5)])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=10, max_consecutive_parse_errors=2)
    agent.add_tools(ADD_TOOL)
    with pytest.raises(AgentEmptyResponseError):
        agent.invoke("confuse me")


def test_native_loop_timeout():
    from autourgos_agent import AgentTimeoutError

    class SlowNativeLLM(ScriptedToolCallLLM):
        def invoke_with_tools(self, prompt, tools, **kwargs):
            time.sleep(0.15)
            return super().invoke_with_tools(prompt, tools, **kwargs)

    llm = SlowNativeLLM([
        ScriptedToolCallLLM.tool_call("add", {"a": 1, "b": 1}, call_id=f"c{i}") for i in range(5)
    ])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=10, max_execution_time=0.1)
    agent.add_tools(ADD_TOOL)
    with pytest.raises(AgentTimeoutError):
        agent.invoke("go")


def test_native_loop_fails_fast_when_llm_lacks_invoke_with_tools():
    from autourgos_agent.testing import ScriptedFakeLLM

    agent = Agent(llm=ScriptedFakeLLM([]), tool_calling_mode="native", max_iterations=3)
    agent.add_tools(ADD_TOOL)
    with pytest.raises(RuntimeError, match="invoke_with_tools"):
        agent.invoke("go")


def test_native_loop_fails_fast_on_notimplementederror():
    from autourgos_agent.base import BaseLLM

    class UnsupportedLLM(BaseLLM):
        def invoke(self, prompt, **kwargs):
            raise NotImplementedError

        async def ainvoke(self, prompt, **kwargs):
            raise NotImplementedError

        def invoke_with_tools(self, prompt, tools, **kwargs):
            raise NotImplementedError("this LLM class doesn't support native tool calling")

    agent = Agent(llm=UnsupportedLLM(), tool_calling_mode="native", max_iterations=3)
    agent.add_tools(ADD_TOOL)
    with pytest.raises(RuntimeError, match="invoke_with_tools"):
        agent.invoke("go")


# -- callbacks ------------------------------------------------------------------------

def test_native_loop_fires_tool_start_and_end_callbacks():
    events: list = []

    class SpyHandler(CallbackHandler):
        def on_tool_start(self, tool_name, tool_input, agent=None, **kwargs):
            events.append(("start", tool_name, tool_input))
        def on_tool_end(self, tool_name, result, agent=None, **kwargs):
            events.append(("end", tool_name, result))

    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("add", {"a": 2, "b": 3}, call_id="c1"),
        ScriptedToolCallLLM.final("5"),
    ])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5, middleware=[SpyHandler()])
    agent.add_tools(ADD_TOOL)
    agent.invoke("2+3?")

    assert ("start", "add", {"a": 2, "b": 3}) in events
    assert ("end", "add", "5") in events


def test_native_loop_never_fires_thought_on_tool_call_turns():
    """
    Documented, intentional limitation (not a bug): invoke_with_tools()
    drops any accompanying reasoning text when the model also returns
    tool_calls, so on_iteration/logger.thought() only ever fire on the
    final-answer turn in native mode, never on a tool-call turn.
    """
    events: list = []

    class SpyHandler(CallbackHandler):
        def on_iteration(self, iteration, thought, agent=None, **kwargs):
            events.append(("thought", iteration, thought))
        def on_agent_end(self, result, agent=None, **kwargs):
            events.append(("end", result))

    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("add", {"a": 2, "b": 3}, call_id="c1"),
        ScriptedToolCallLLM.final("5"),
    ])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5, middleware=[SpyHandler()])
    agent.add_tools(ADD_TOOL)
    agent.invoke("2+3?")

    assert not any(e[0] == "thought" for e in events)
    assert ("end", "5") in events
