"""
Sprint 3 (FRAMEWORK_REVIEW.md Finding #3) regression coverage.

  - An Agent instance now enforces one active run at a time: a second
    invoke()/ainvoke() call while a run is already in progress on that same
    instance raises AgentAlreadyRunningError instead of silently clobbering
    shared mid-run state (current_query, scratchpad). Sequential reuse of
    one instance (non-overlapping calls) is unaffected, including after a
    prior run raised.
  - max_tool_workers now genuinely caps concurrent tool execution on the
    ASYNC path too (previously a plain, unbounded asyncio.gather -- the cap
    only applied to the sync/ThreadPoolExecutor path).
  - The sync path's ThreadPoolExecutor is now created once per run and
    reused across iterations, instead of a fresh pool every iteration.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from autourgos_agent import Agent, AgentAlreadyRunningError, AgentTimeoutError
from autourgos_agent.testing import ScriptedFakeLLM


def _tool(name: str, func, description: str = "d"):
    return {"name": name, "description": description, "parameters": {}, "func": func}


def _multi_tool_call_responses(n: int):
    return [
        json.dumps({
            "thought": "run tools",
            "actions": [{"action": f"t{i}", "action_input": {"n": i}} for i in range(n)],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
    ]


class _SlowLLM:
    """Never finishes quickly -- gives a concurrent second call time to
    observe _run_active still set before the first call returns."""

    def __init__(self, delay: float = 0.2) -> None:
        self.delay = delay

    def invoke(self, prompt, **kwargs):
        time.sleep(self.delay)
        return json.dumps({"thought": None, "actions": [], "final_answer": "done"})

    async def ainvoke(self, prompt, **kwargs):
        await asyncio.sleep(self.delay)
        return json.dumps({"thought": None, "actions": [], "final_answer": "done"})


# -- one active run per instance ------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_ainvoke_on_same_instance_raises_already_running():
    agent = Agent(llm=_SlowLLM(0.2), max_iterations=5)

    async def _second_call_soon():
        await asyncio.sleep(0.02)  # let the first call acquire the guard
        with pytest.raises(AgentAlreadyRunningError):
            await agent.ainvoke("second")

    await asyncio.gather(agent.ainvoke("first"), _second_call_soon())


def test_concurrent_invoke_on_same_instance_raises_already_running():
    agent = Agent(llm=_SlowLLM(0.2), max_iterations=5)
    errors = []

    def _second_call_soon():
        time.sleep(0.02)
        try:
            agent.invoke("second")
        except AgentAlreadyRunningError as exc:
            errors.append(exc)

    t = threading.Thread(target=_second_call_soon)
    t.start()
    result = agent.invoke("first")
    t.join()

    assert result == "done"
    assert len(errors) == 1


def test_sequential_reuse_of_same_instance_still_works():
    """Non-overlapping calls on one instance must be unaffected."""
    agent = Agent(llm=ScriptedFakeLLM([
        json.dumps({"thought": None, "actions": [], "final_answer": "one"}),
        json.dumps({"thought": None, "actions": [], "final_answer": "two"}),
    ]), max_iterations=5)

    assert agent.invoke("a") == "one"
    assert agent.invoke("b") == "two"


def test_run_guard_clears_after_a_run_that_raises():
    """A run that raises (e.g. timeout) must still clear the guard so a
    later call on the same instance can succeed."""
    agent = Agent(llm=_SlowLLM(0.2), max_iterations=5, max_execution_time=0.01)

    with pytest.raises(AgentTimeoutError):
        agent.invoke("times out")

    # Reuse the SAME agent instance whose prior run raised.
    agent.llm = ScriptedFakeLLM([
        json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
    ])
    agent.max_execution_time = None
    assert agent.invoke("now succeeds") == "done"


# -- bounded async tool concurrency ----------------------------------------------

@pytest.mark.asyncio
async def test_async_tool_execution_respects_max_tool_workers_cap():
    """FRAMEWORK_REVIEW.md Finding #3: max_tool_workers=1 previously had no
    effect on the async path -- asyncio.gather ran every approved tool call
    with no cap at all. Assert the observed in-flight count never exceeds
    the configured cap, not just that timing looks serial (timing alone
    can't distinguish 'capped at 1' from 'capped at 2' etc.)."""
    in_flight = 0
    max_observed = 0
    lock = asyncio.Lock()

    async def tracked_tool(n: int) -> int:
        nonlocal in_flight, max_observed
        async with lock:
            in_flight += 1
            max_observed = max(max_observed, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return n

    tools = [_tool(f"t{i}", tracked_tool) for i in range(4)]
    agent = Agent(
        llm=ScriptedFakeLLM(_multi_tool_call_responses(4)),
        max_iterations=5,
        max_tool_workers=1,
    )
    agent.add_tools(*tools)

    result = await agent.ainvoke("run 4 tools capped at 1")

    assert result == "done"
    assert max_observed == 1


def test_sync_tool_execution_respects_max_tool_workers_cap():
    in_flight = 0
    max_observed = 0
    lock = threading.Lock()

    def tracked_tool(n: int) -> int:
        nonlocal in_flight, max_observed
        with lock:
            in_flight += 1
            max_observed = max(max_observed, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return n

    tools = [_tool(f"t{i}", tracked_tool) for i in range(4)]
    agent = Agent(
        llm=ScriptedFakeLLM(_multi_tool_call_responses(4)),
        max_iterations=5,
        max_tool_workers=2,
    )
    agent.add_tools(*tools)

    result = agent.invoke("run 4 tools capped at 2")

    assert result == "done"
    assert max_observed <= 2


@pytest.mark.asyncio
async def test_async_agent_sync_tool_does_not_block_event_loop():
    """T-001: a blocking (plain, non-async) tool func called from the async
    loop must run on a worker thread, not inline on the event loop -- a
    concurrent heartbeat task should keep making progress while it runs,
    instead of freezing for the tool's whole duration."""
    def blocking_tool() -> str:
        time.sleep(0.3)
        return "done"

    heartbeat_ticks = []

    async def _heartbeat():
        while len(heartbeat_ticks) < 5:
            await asyncio.sleep(0.02)
            heartbeat_ticks.append(time.monotonic())

    agent = Agent(
        llm=ScriptedFakeLLM([
            json.dumps({
                "thought": "call it",
                "actions": [{"action": "blocking_tool", "action_input": {}}],
                "final_answer": None,
            }),
            json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
        ]),
        max_iterations=5,
    )
    agent.add_tools(_tool("blocking_tool", blocking_tool))

    start = time.monotonic()
    heartbeat_task = asyncio.ensure_future(_heartbeat())
    result = await agent.ainvoke("run blocking tool")
    await heartbeat_task

    assert result == "done"
    # If the event loop were blocked for the tool's whole 0.3s call, the
    # heartbeat couldn't get its first ~0.02s slice until after that --
    # assert it actually interleaved, not just that it eventually ran 5
    # times after the blocking call returned control to the loop.
    assert heartbeat_ticks[0] - start < 0.15


def test_sync_loop_reuses_one_thread_pool_executor_across_iterations(monkeypatch):
    """Previously a brand-new ThreadPoolExecutor was constructed every
    iteration that had approved tool calls. Over a multi-iteration run this
    meant MAX_TOOL_WORKERS only ever capped one iteration's tool calls, not
    the whole run's -- assert the constructor is only called once."""
    construct_count = 0
    real_init = ThreadPoolExecutor.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal construct_count
        construct_count += 1
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "__init__", counting_init)

    responses = [
        json.dumps({"thought": None, "actions": [{"action": "noop", "action_input": {}}], "final_answer": None})
        for _ in range(3)
    ] + [json.dumps({"thought": None, "actions": [], "final_answer": "done"})]
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=10)
    agent.add_tools(_tool("noop", lambda: "ok"))

    result = agent.invoke("run several iterations with tool calls")

    assert result == "done"
    assert construct_count == 1
