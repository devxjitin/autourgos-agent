"""
Coverage for Agent(pre_iteration_callback=..., pre_iteration_files=...) --
the inline (non-middleware) replacement for manually wiring
PreIterationMiddleware via middleware=[...]. See _preiteration.py's
_PreIterationRuntime docstring for why this exists as a separate,
flat-state path instead of going through the CallbackHandler bus.
"""
import json
import os
import tempfile

import pytest

from autourgos_agent import Agent
from autourgos_agent.testing import make_test_agent


def _final(text: str) -> str:
    return json.dumps({"thought": None, "actions": [], "final_answer": text})


@pytest.fixture
def temp_file():
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(path, "w") as f:
        f.write("hello")
    yield path
    try:
        os.remove(path)
    except OSError:
        pass


def test_default_agent_has_null_preiteration_runtime():
    """No pre_iteration_callback=/pre_iteration_files= -> the zero-cost
    no-op stand-in, not a real runtime instance."""
    from autourgos_agent._preiteration import _NULL_PREITERATION

    agent = make_test_agent(responses=[_final("done")])
    assert agent._preiteration is _NULL_PREITERATION


def test_pre_iteration_callback_runs_every_iteration_sync():
    calls = []
    agent = make_test_agent(
        responses=[_final("done")],
        pre_iteration_callback=lambda i: calls.append(i),
    )
    result = agent.invoke("go")

    assert result == "done"
    assert calls == [1]


@pytest.mark.asyncio
async def test_pre_iteration_callback_runs_every_iteration_async():
    calls = []
    agent = make_test_agent(
        responses=[_final("done")],
        pre_iteration_callback=lambda i: calls.append(i),
    )
    result = await agent.ainvoke("go")

    assert result == "done"
    assert calls == [1]


def test_pre_iteration_files_injected_into_llm_call_sync(temp_file):
    agent = make_test_agent(responses=[_final("done")], pre_iteration_files=temp_file)
    agent.invoke("go")

    assert agent.llm.calls[0]["kwargs"]["files"] == [temp_file]


@pytest.mark.asyncio
async def test_pre_iteration_files_injected_into_llm_call_async(temp_file):
    agent = make_test_agent(responses=[_final("done")], pre_iteration_files=temp_file)
    await agent.ainvoke("go")

    assert agent.llm.calls[0]["kwargs"]["files"] == [temp_file]


def test_pre_iteration_files_callable_resolves_per_iteration(temp_file):
    agent = make_test_agent(
        responses=[_final("done")],
        pre_iteration_files=lambda iteration: temp_file,
    )
    agent.invoke("go")

    assert agent.llm.calls[0]["kwargs"]["files"] == [temp_file]


def test_pre_iteration_callback_error_does_not_crash_run():
    def bad_callback(iteration):
        raise RuntimeError("boom")

    agent = make_test_agent(responses=[_final("done")], pre_iteration_callback=bad_callback)
    result = agent.invoke("go")

    assert result == "done"


def test_invalid_image_quality_int_rejected():
    with pytest.raises(ValueError, match="image_quality"):
        Agent(pre_iteration_files="x.png", image_quality=999)


def test_invalid_image_quality_string_rejected():
    with pytest.raises(ValueError, match="image_quality"):
        Agent(pre_iteration_files="x.png", image_quality="ultra")


def test_other_middleware_before_iteration_kwargs_take_precedence(temp_file):
    """cb.fire_before_iteration()'s result must win on key conflicts over
    the inline preiteration kwargs, matching "later handlers override
    earlier" semantics."""
    from autourgos_agent import CallbackHandler

    class OverridingMiddleware(CallbackHandler):
        def on_before_iteration(self, iteration, agent=None, **kwargs):
            return {"files": ["override.png"]}

    agent = make_test_agent(
        responses=[_final("done")],
        pre_iteration_files=temp_file,
        middleware=[OverridingMiddleware()],
    )
    agent.invoke("go")

    assert agent.llm.calls[0]["kwargs"]["files"] == ["override.png"]
