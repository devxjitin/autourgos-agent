"""
FRAMEWORK_REVIEW.md Finding #2 regression tests: Agent.invoke()/ainvoke() used
to catch only `Exception`, so a cancelled async run (asyncio.CancelledError,
a BaseException subclass since Python 3.8) or a sync KeyboardInterrupt/
SystemExit bypassed on_agent_error entirely -- and every real middleware
package (skills, toolbox, hcix, history, preiteration) does its cleanup
(removing injected tools/prompt blocks, stopping listeners, flushing logs,
deleting temp files) exclusively inside on_agent_error/on_agent_end. A
cancelled run left that cleanup permanently skipped.
"""
from __future__ import annotations

import asyncio

import pytest

from autourgos_agent import Agent, CallbackHandler


class _CleanupTrackingMiddleware(CallbackHandler):
    """Mirrors the real middleware pattern: add a tool in on_agent_start,
    remove it in on_agent_end/on_agent_error -- exactly like SkillLibrary/
    ToolboxMiddleware's load_skill/expose_toolbox meta-tools."""

    def __init__(self) -> None:
        self.error_calls = []
        self._added_tool = {"name": "injected_tool", "description": "d", "parameters": {}, "func": lambda: None}

    def on_agent_start(self, query, agent=None, **kwargs):
        agent.add_tools(self._added_tool)

    def on_agent_error(self, error, agent=None, **kwargs):
        self.error_calls.append(error)
        agent.tools = [t for t in agent.tools if t is not self._added_tool]

    def on_agent_end(self, response, agent=None, **kwargs):
        agent.tools = [t for t in agent.tools if t is not self._added_tool]


def _tool_names(agent):
    return [t["name"] for t in agent.tools]


@pytest.mark.asyncio
async def test_cancelled_ainvoke_still_fires_on_agent_error_cleanup():
    class HangingLLM:
        def invoke(self, prompt, **kwargs):
            raise NotImplementedError

        async def ainvoke(self, prompt, **kwargs):
            await asyncio.sleep(10)
            return '{"thought": null, "actions": [], "final_answer": "done"}'

    middleware = _CleanupTrackingMiddleware()
    agent = Agent(llm=HangingLLM(), max_iterations=5, middleware=[middleware])

    task = asyncio.ensure_future(agent.ainvoke("go"))
    await asyncio.sleep(0.05)
    assert "injected_tool" in _tool_names(agent)  # on_agent_start ran, tool present mid-flight

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(middleware.error_calls) == 1
    assert isinstance(middleware.error_calls[0], asyncio.CancelledError)
    assert "injected_tool" not in _tool_names(agent)  # cleanup ran despite cancellation


def test_keyboard_interrupt_in_sync_loop_still_fires_on_agent_error_cleanup():
    class InterruptingLLM:
        def invoke(self, prompt, **kwargs):
            raise KeyboardInterrupt

        async def ainvoke(self, prompt, **kwargs):
            return self.invoke(prompt, **kwargs)

    middleware = _CleanupTrackingMiddleware()
    agent = Agent(llm=InterruptingLLM(), max_iterations=5, middleware=[middleware])

    with pytest.raises(KeyboardInterrupt):
        agent.invoke("go")

    assert len(middleware.error_calls) == 1
    assert isinstance(middleware.error_calls[0], KeyboardInterrupt)
    assert "injected_tool" not in _tool_names(agent)


def test_normal_agent_error_path_unaffected():
    """Regression guard: widening except Exception -> except BaseException
    must not change behavior for an ordinary Exception."""
    from autourgos_agent import AgentLLMError

    class RaisingLLM:
        def invoke(self, prompt, **kwargs):
            raise RuntimeError("boom")

        async def ainvoke(self, prompt, **kwargs):
            return self.invoke(prompt, **kwargs)

    middleware = _CleanupTrackingMiddleware()
    agent = Agent(llm=RaisingLLM(), max_iterations=5, middleware=[middleware])

    with pytest.raises(AgentLLMError):
        agent.invoke("go")

    assert len(middleware.error_calls) == 1
    assert isinstance(middleware.error_calls[0], AgentLLMError)
