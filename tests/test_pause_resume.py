"""
Tests for Agent.pause()/resume()/is_paused -- in-process blocking pause,
effective at the next iteration boundary in all 4 loop variants (sync/async
x prompt/native mode). See README's "Pause & Resume" section.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, List, Optional, Tuple

import pytest

from autourgos_agent import Agent, CallbackHandler
from autourgos_agent.testing import ScriptedToolCallLLM, make_test_agent


def _final_response(answer: str) -> str:
    return json.dumps({"thought": None, "actions": [], "final_answer": answer})


# -- pre-invoke pause (deterministic: blocks before the very first iteration) --

def test_pause_before_invoke_blocks_sync_run_until_resumed():
    agent = make_test_agent(responses=[_final_response("done")])
    agent.pause(reason="pre-invoke")
    assert agent.is_paused

    result_holder: dict = {}

    def run():
        result_holder["result"] = agent.invoke("hi")

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.15)
    # Still blocked -- nothing has run yet.
    assert "result" not in result_holder
    assert agent.is_paused

    agent.resume()
    t.join(timeout=5)
    assert result_holder["result"] == "done"
    assert not agent.is_paused


@pytest.mark.asyncio
async def test_pause_before_ainvoke_blocks_async_run_until_resumed():
    agent = make_test_agent(responses=[_final_response("done")])
    agent.pause(reason="pre-invoke")

    task = asyncio.ensure_future(agent.ainvoke("hi"))
    await asyncio.sleep(0.1)
    assert not task.done()
    assert agent.is_paused

    agent.resume()
    result = await asyncio.wait_for(task, timeout=5)
    assert result == "done"


def test_pause_before_invoke_blocks_native_mode_sync_run():
    llm = ScriptedToolCallLLM([ScriptedToolCallLLM.final("done")])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5)
    agent.add_tools()
    agent.pause(reason="native-pre-invoke")

    result_holder: dict = {}

    def run():
        result_holder["result"] = agent.invoke("hi")

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.15)
    assert "result" not in result_holder

    agent.resume()
    t.join(timeout=5)
    assert result_holder["result"] == "done"


@pytest.mark.asyncio
async def test_pause_before_ainvoke_blocks_native_mode_async_run():
    llm = ScriptedToolCallLLM([ScriptedToolCallLLM.final("done")])
    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5)
    agent.add_tools()
    agent.pause(reason="native-pre-ainvoke")

    task = asyncio.ensure_future(agent.ainvoke("hi"))
    await asyncio.sleep(0.1)
    assert not task.done()

    agent.resume()
    result = await asyncio.wait_for(task, timeout=5)
    assert result == "done"


# -- pause triggered from inside a middleware hook (external-signal pattern,
# but issued from the same thread the loop calls the hook on) -----------------

class PauseOnFirstIteration(CallbackHandler):
    """Pauses the agent the first time on_iteration_start fires. A separate
    thread is expected to call resume() shortly after, unblocking the loop
    from the SAME on_iteration_start call that requested the pause."""

    def __init__(self) -> None:
        self.paused_once = False

    def on_iteration_start(self, iteration: int, agent: Any = None, **kwargs: Any) -> None:
        if not self.paused_once:
            self.paused_once = True
            agent.pause(reason="middleware")


def test_middleware_triggered_pause_blocks_until_external_resume():
    responses = [_final_response("done")]
    mw = PauseOnFirstIteration()
    agent = make_test_agent(responses=responses, middleware=[mw])

    def resume_later():
        time.sleep(0.2)
        assert agent.is_paused
        agent.resume()

    t = threading.Thread(target=resume_later)
    t.start()

    start = time.monotonic()
    result = agent.invoke("hi")
    elapsed = time.monotonic() - start
    t.join(timeout=5)

    assert result == "done"
    # Loose tolerance -- thread-scheduling jitter (esp. on Windows) can wake
    # resume_later() a few ms before the full 0.2s elapses, which isn't a
    # real assertion failure, just timer granularity.
    assert elapsed >= 0.15
    assert not agent.is_paused


# -- resume() without a prior pause() is a no-op -------------------------------

def test_resume_without_pause_is_noop():
    agent = make_test_agent(responses=[_final_response("done")])
    agent.resume()  # must not hang or raise
    assert not agent.is_paused
    result = agent.invoke("hi")
    assert result == "done"


# -- max_execution_time excludes time spent paused -----------------------------

def test_paused_duration_excluded_from_max_execution_time_deadline():
    agent = make_test_agent(
        responses=[_final_response("done")],
        max_execution_time=0.3,
    )
    agent.pause(reason="deadline-test")

    def resume_later():
        # Longer than max_execution_time -- if paused time weren't excluded,
        # resuming would immediately blow the deadline and raise
        # AgentTimeoutError instead of completing normally.
        time.sleep(0.5)
        agent.resume()

    t = threading.Thread(target=resume_later)
    t.start()
    result = agent.invoke("hi")
    t.join(timeout=5)

    assert result == "done"


# -- on_agent_pause / on_agent_resume hooks fire -------------------------------

class PauseResumeRecorder(CallbackHandler):
    def __init__(self) -> None:
        self.events: List[Tuple[str, int, Any]] = []

    def on_agent_pause(self, iteration: int, reason: Optional[str], agent: Any = None, **kwargs: Any) -> None:
        self.events.append(("pause", iteration, reason))

    def on_agent_resume(self, iteration: int, paused_duration: float, agent: Any = None, **kwargs: Any) -> None:
        self.events.append(("resume", iteration, paused_duration >= 0))


def test_on_agent_pause_and_resume_hooks_fire_with_expected_args():
    recorder = PauseResumeRecorder()
    agent = make_test_agent(responses=[_final_response("done")], middleware=[recorder])
    agent.pause(reason="hook-test")

    def resume_later():
        time.sleep(0.05)
        agent.resume()

    t = threading.Thread(target=resume_later)
    t.start()
    agent.invoke("hi")
    t.join(timeout=5)

    assert recorder.events == [("pause", 1, "hook-test"), ("resume", 1, True)]
