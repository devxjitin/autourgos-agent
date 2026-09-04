"""
Regression tests for two related CallbackManager/AgentLogger bugs:

1. AgentLogger had no .warning() method, despite being duck-typed as a
   logger-shaped object by middleware (e.g. autourgos-hcix's
   CognitiveInterruptManager.poll(logger=agent.logger)) that calls
   .warning() on it -- crashing with AttributeError, silently swallowed by
   CallbackManager since poll() runs from a hook.

2. CallbackManager._call_with_agent_fallback (and the three duplicated
   on_before_iteration call sites) decided whether a handler accepts
   agent= by calling it and retrying on TypeError -- so a handler whose
   OWN body raised an unrelated TypeError got silently called AGAIN,
   doubling any real side effect for that one event.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from autourgos_agent import Agent, AgentLogger, CallbackHandler, CallbackManager


class FakeLLM:
    def invoke(self, prompt: Any, **kwargs: Any) -> str:
        return json.dumps({"thought": None, "actions": [], "final_answer": "done"})

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> str:
        return self.invoke(prompt, **kwargs)


# -- (1) AgentLogger.warning() -------------------------------------------------

def test_agent_logger_has_warning_method(capsys) -> None:
    logger = AgentLogger(verbose=True)
    logger.warning("something to warn about")
    out = capsys.readouterr().out
    assert "Warning: something to warn about" in out


def test_agent_logger_warning_silent_when_not_verbose(capsys) -> None:
    logger = AgentLogger(verbose=False)
    logger.warning("quiet warning")
    out = capsys.readouterr().out
    assert out == ""


# -- (2) no double-call on an internal TypeError -------------------------------

def test_handler_internal_type_error_does_not_retry_and_double_fire() -> None:
    """A handler whose body raises TypeError for reasons unrelated to its
    signature (e.g. it does `1 + "x"` internally) must be called exactly
    once for that event -- not twice."""
    calls: List[int] = []

    class BuggyHandler(CallbackHandler):
        def on_agent_start(self, query: str, agent: Any = None, **kwargs: Any) -> None:
            calls.append(1)
            raise TypeError("unrelated bug inside my handler, nothing to do with agent=")

    agent = Agent(llm=FakeLLM(), middleware=[BuggyHandler()])
    agent.invoke("hi")  # CallbackManager swallows handler exceptions; must not raise

    assert len(calls) == 1


def test_narrow_handler_without_agent_param_still_works_sync() -> None:
    """A genuinely old-style handler (no `agent` param, no `**kwargs`)
    must still receive the call, with `agent=` correctly omitted --
    covering the actual signature-mismatch case _accepts_agent_kwarg
    exists for."""
    calls: List[str] = []

    class NarrowHandler(CallbackHandler):
        def on_agent_start(self, query: str) -> None:
            calls.append(query)

    agent = Agent(llm=FakeLLM(), middleware=[NarrowHandler()])
    agent.invoke("narrow-sync")

    assert calls == ["narrow-sync"]


def test_narrow_handler_without_agent_param_still_works_async() -> None:
    calls: List[str] = []

    class NarrowHandler(CallbackHandler):
        def on_agent_start(self, query: str) -> None:
            calls.append(query)

    agent = Agent(llm=FakeLLM(), middleware=[NarrowHandler()])
    asyncio.run(agent.ainvoke("narrow-async"))

    assert calls == ["narrow-async"]


def test_on_before_iteration_internal_type_error_does_not_double_fire_sync() -> None:
    calls: List[int] = []

    class BuggyBeforeIteration(CallbackHandler):
        def on_before_iteration(self, iteration: int, agent: Any = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
            calls.append(iteration)
            raise TypeError("bug inside on_before_iteration, unrelated to agent=")

    agent = Agent(llm=FakeLLM(), middleware=[BuggyBeforeIteration()])
    agent.invoke("hi")

    assert calls == [1]


def test_on_before_iteration_internal_type_error_does_not_double_fire_async_offloaded() -> None:
    """Exercises afire_before_iteration's sync-offload branch (run through
    the executor) specifically."""
    calls: List[int] = []

    class BuggyBeforeIteration(CallbackHandler):
        def on_before_iteration(self, iteration: int, agent: Any = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
            calls.append(iteration)
            raise TypeError("bug inside on_before_iteration, unrelated to agent=")

    agent = Agent(llm=FakeLLM(), middleware=[BuggyBeforeIteration()])
    asyncio.run(agent.ainvoke("hi"))

    assert calls == [1]


def test_on_before_iteration_internal_type_error_does_not_double_fire_async_coroutine() -> None:
    """Exercises afire_before_iteration's `inspect.iscoroutinefunction(fn)`
    branch specifically (an async on_before_iteration handler)."""
    calls: List[int] = []

    class BuggyAsyncBeforeIteration(CallbackHandler):
        async def on_before_iteration(self, iteration: int, agent: Any = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
            calls.append(iteration)
            raise TypeError("bug inside async on_before_iteration, unrelated to agent=")

    agent = Agent(llm=FakeLLM(), middleware=[BuggyAsyncBeforeIteration()])
    asyncio.run(agent.ainvoke("hi"))

    assert calls == [1]


def test_narrow_on_before_iteration_without_agent_param_still_merges_kwargs() -> None:
    """A genuinely narrow on_before_iteration (no agent param, no
    **kwargs) still gets called and its returned dict still merges into
    the per-iteration LLM call kwargs, matching documented behavior."""

    class NarrowBeforeIteration(CallbackHandler):
        def on_before_iteration(self, iteration: int) -> Optional[Dict[str, Any]]:
            return {"injected": "value"}

    seen_kwargs: List[Dict[str, Any]] = []

    class SpyLLM:
        def invoke(self, prompt: Any, **kwargs: Any) -> str:
            seen_kwargs.append(kwargs)
            return json.dumps({"thought": None, "actions": [], "final_answer": "done"})

        async def ainvoke(self, prompt: Any, **kwargs: Any) -> str:
            return self.invoke(prompt, **kwargs)

    agent = Agent(llm=SpyLLM(), middleware=[NarrowBeforeIteration()])
    agent.invoke("hi")

    assert seen_kwargs and seen_kwargs[0].get("injected") == "value"
