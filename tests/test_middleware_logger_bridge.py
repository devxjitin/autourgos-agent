"""
Tests for AgentLogger.middleware() -- the narrative bridge that lets
middleware/callback handlers (toolbox, summarizer, hcix, etc.) print into
the same verbose trace as the core agent loop.
"""

from __future__ import annotations

import io
import json
import re
from contextlib import redirect_stdout
from typing import Any, List

from autourgos_agent import Agent, CallbackHandler


class FakeLLM:
    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)

    def invoke(self, prompt: Any, **kwargs: Any) -> str:
        return self._responses.pop(0)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


class NarratingMiddleware(CallbackHandler):
    """Mimics how a real middleware (e.g. toolbox) would narrate an action."""

    def on_agent_start(self, query, agent=None, **kwargs):
        logger = getattr(agent, "logger", None)
        if logger:
            logger.middleware("Toolbox", "Exposed toolbox 'search_tools' to agent.")

    def on_iteration_start(self, iteration, agent=None, **kwargs):
        logger = getattr(agent, "logger", None)
        if logger and iteration == 1:
            logger.middleware("Summarizer", "Compressed scratchpad (iteration 1).")


def test_middleware_bridge_prints_with_source_prefix_when_verbose():
    responses = [json.dumps({"thought": None, "actions": [], "final_answer": "done"})]
    agent = Agent(llm=FakeLLM(responses), verbose=True, middleware=[NarratingMiddleware()])
    agent.add_tools({"name": "noop", "description": "no-op", "func": lambda: "ok"})

    buf = io.StringIO()
    with redirect_stdout(buf):
        agent.invoke("go")

    output = _strip_ansi(buf.getvalue())
    assert "[Toolbox] Exposed toolbox 'search_tools' to agent." in output
    assert "[Summarizer] Compressed scratchpad (iteration 1)." in output


def test_middleware_bridge_silent_when_verbose_false():
    responses = [json.dumps({"thought": None, "actions": [], "final_answer": "done"})]
    agent = Agent(llm=FakeLLM(responses), verbose=False, middleware=[NarratingMiddleware()])
    agent.add_tools({"name": "noop", "description": "no-op", "func": lambda: "ok"})

    buf = io.StringIO()
    with redirect_stdout(buf):
        agent.invoke("go")

    assert buf.getvalue() == ""


def test_middleware_bridge_does_not_crash_when_agent_has_no_logger():
    class FakeAgentNoLogger:
        pass

    class DirectHandler(CallbackHandler):
        def on_agent_start(self, query, agent=None, **kwargs):
            logger = getattr(agent, "logger", None)
            if logger:
                logger.middleware("Toolbox", "should not print")

    handler = DirectHandler()
    fake_agent = FakeAgentNoLogger()
    # Should not raise even though fake_agent has no .logger attribute.
    handler.on_agent_start("q", agent=fake_agent)


def test_agent_logger_middleware_method_directly():
    from autourgos_agent.logging import AgentLogger

    logger = AgentLogger(verbose=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        logger.middleware("HCIx", "Human override injected.")

    output = _strip_ansi(buf.getvalue())
    assert output.strip() == "[HCIx] Human override injected."
