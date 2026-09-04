"""
Regression test: CallbackManager._afire / afire_before_iteration offload sync
hooks to a ThreadPoolExecutor via loop.run_in_executor(), which by itself
does NOT propagate ContextVar writes made inside the worker thread back to
later hook calls in the same run (this holds for run_in_executor AND for
asyncio.to_thread -- both create a fresh contextvars.Context snapshot per
call unless the caller explicitly reuses one Context object across calls).

A handler keeping per-run state in a ContextVar (e.g. autourgos-history's
AgentHistoryMiddleware) needs writes made in one hook call to be visible to
reads made in a LATER hook call within the same ainvoke() run -- and needs
that isolated from other concurrent ainvoke() runs sharing the same
CallbackManager instance.

Fixed via CallbackManager.capture_run_context(), called once at the top of
Agent.ainvoke(): it snapshots the calling task's Context and stores it in
a ContextVar (_run_context_var) so every subsequent sync-hook offload for
that run reuses the SAME Context object (contextvars.Context.run() only
makes writes visible to a LATER .run() call on that same object -- a fresh
copy per call, the naive/wrong fix, does not).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
from typing import Any, List

from autourgos_agent import Agent, CallbackHandler

_run_id: "contextvars.ContextVar[str]" = contextvars.ContextVar("run_id", default=None)


class FakeLLM:
    def invoke(self, prompt: Any, **kwargs: Any) -> str:
        return json.dumps({"thought": None, "actions": [], "final_answer": "done"})

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> str:
        return self.invoke(prompt, **kwargs)


class ToolLLM:
    """Calls one tool, then finishes -- so tool hooks fire too."""

    def __init__(self) -> None:
        self.n = 0

    def invoke(self, prompt: Any, **kwargs: Any) -> str:
        self.n += 1
        if self.n == 1:
            return json.dumps(
                {"thought": None, "actions": [{"action": "noop", "action_input": {}}], "final_answer": None}
            )
        return json.dumps({"thought": None, "actions": [], "final_answer": "done"})

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> str:
        return self.invoke(prompt, **kwargs)


def noop_tool() -> str:
    return "ok"


class ContextVarHandler(CallbackHandler):
    """Sets a ContextVar in on_agent_start (sync hook, runs off-thread under
    ainvoke()) and reads it back in on_agent_end (also sync, also
    off-thread). Records what it saw so the test can assert the write was
    actually visible on read, not silently isolated into a throwaway
    context copy."""

    def __init__(self) -> None:
        self.seen_on_end: List[Any] = []

    def on_agent_start(self, query: str, agent: Any = None, **kwargs: Any) -> None:
        _run_id.set(query)

    def on_agent_end(self, result: str, agent: Any = None, **kwargs: Any) -> None:
        self.seen_on_end.append(_run_id.get())


def test_sync_hook_contextvar_write_visible_to_later_sync_hook_under_ainvoke() -> None:
    handler = ContextVarHandler()
    agent = Agent(llm=FakeLLM(), middleware=[handler])

    asyncio.run(agent.ainvoke("run-A"))

    assert handler.seen_on_end == ["run-A"]


def test_concurrent_ainvoke_runs_do_not_leak_contextvar_state() -> None:
    handler = ContextVarHandler()
    agent = Agent(llm=FakeLLM(), middleware=[handler])

    async def _main() -> None:
        await asyncio.gather(
            agent.ainvoke("run-A"),
            agent.ainvoke("run-B"),
        )

    asyncio.run(_main())

    assert sorted(handler.seen_on_end) == ["run-A", "run-B"]


def test_contextvar_write_visible_across_concurrent_tool_hook_fanout() -> None:
    """Tool hooks for concurrently-executed tools run inside their own
    asyncio.gather()-created Tasks -- confirms the reused run Context is
    still visible (by reference) inside those nested tasks, not just at the
    top level of the run."""

    class ToolHookHandler(CallbackHandler):
        def __init__(self) -> None:
            self.seen_in_tool_end: List[Any] = []

        def on_agent_start(self, query: str, agent: Any = None, **kwargs: Any) -> None:
            _run_id.set(query)

        def on_tool_end(self, tool_name: str, result: str, agent: Any = None, **kwargs: Any) -> None:
            self.seen_in_tool_end.append(_run_id.get())

    handler = ToolHookHandler()
    agent = Agent(llm=ToolLLM(), tools=[{"name": "noop", "description": "", "parameters": {}, "func": noop_tool}],
                  middleware=[handler])

    asyncio.run(agent.ainvoke("tool-run"))

    assert handler.seen_in_tool_end == ["tool-run"]


def test_on_before_iteration_contextvar_write_visible_under_ainvoke() -> None:
    """Same propagation guarantee for afire_before_iteration's two
    run_in_executor call sites (agent= accepted, and the TypeError-retry
    fallback for narrower handler signatures)."""

    class BeforeIterHandler(CallbackHandler):
        def __init__(self) -> None:
            self.seen: List[Any] = []

        def on_before_iteration(self, iteration: int, agent: Any = None, **kwargs: Any):
            _run_id.set(f"iter-{iteration}")
            return None

        def on_agent_end(self, result: str, agent: Any = None, **kwargs: Any) -> None:
            self.seen.append(_run_id.get())

    class NarrowBeforeIterHandler(CallbackHandler):
        """Older-style handler: on_before_iteration doesn't accept agent=,
        exercising the TypeError-retry branch specifically."""

        def __init__(self) -> None:
            self.seen: List[Any] = []

        def on_before_iteration(self, iteration: int):
            _run_id.set(f"narrow-iter-{iteration}")
            return None

        def on_agent_end(self, result: str, agent: Any = None, **kwargs: Any) -> None:
            self.seen.append(_run_id.get())

    h1 = BeforeIterHandler()
    agent1 = Agent(llm=FakeLLM(), middleware=[h1])
    asyncio.run(agent1.ainvoke("q1"))
    assert h1.seen == ["iter-1"]

    h2 = NarrowBeforeIterHandler()
    agent2 = Agent(llm=FakeLLM(), middleware=[h2])
    asyncio.run(agent2.ainvoke("q1"))
    assert h2.seen == ["narrow-iter-1"]


def test_direct_callback_manager_use_without_capture_falls_back_to_old_behavior() -> None:
    """A caller driving CallbackManager directly (never calling
    capture_run_context(), i.e. not going through Agent.ainvoke()) must see
    identical behavior to before this fix: hooks still fire, just without
    ContextVar reuse across calls."""
    from autourgos_agent import CallbackManager

    calls: List[str] = []

    class PlainHandler(CallbackHandler):
        def on_agent_start(self, query: str, agent: Any = None, **kwargs: Any) -> None:
            calls.append(f"start:{query}")

    async def _main() -> None:
        cb = CallbackManager([PlainHandler()])
        await cb.afire_agent_start("hello")

    asyncio.run(_main())
    assert calls == ["start:hello"]
