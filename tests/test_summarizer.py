"""
Tests for Agent's built-in scratchpad summarization -- summarize_every=,
summarizer_llm=, max_scratchpad_chars= constructor kwargs.

Implemented inline in the agent loop (AgentLoopMixin._maybe_summarize /
_amaybe_summarize / _summarize_should_trigger / _do_summarize in base.py),
NOT as a CallbackHandler/middleware -- there is no separate
AutoSummarizeMiddleware class anymore (retired; see CHANGELOG). Regression
coverage below is carried over from that former standalone package/class.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

from autourgos_agent import Agent
from autourgos_agent.testing import ScriptedFakeLLM, make_test_agent


class FakeLLM:
    """A minimal dedicated summarizer_llm -- records call count, ignores the
    prompt, returns a fixed response."""

    def __init__(self, response: str = "condensed summary") -> None:
        self.response = response
        self.calls = 0

    def invoke(self, prompt: str) -> str:
        self.calls += 1
        return self.response


_ECHO_TOOL = {
    "name": "echo",
    "description": "Echo the given text back.",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    "func": lambda text="": f"echo: {text}",
}


def _final_response(answer: str) -> str:
    return json.dumps({"thought": None, "actions": [], "final_answer": answer})


def _make_agent(summarize_every: int = 5, **kwargs) -> Agent:
    """A real Agent with summarize_every set, whose main llm is a harmless
    single-final-answer stub -- used by tests that drive
    _summarize_should_trigger()/_do_summarize() directly instead of going
    through a full invoke() loop."""
    llm = ScriptedFakeLLM([_final_response("n/a")])
    return Agent(llm=llm, summarize_every=summarize_every, **kwargs)


# -- unit-level: drive _summarize_should_trigger()/_do_summarize() directly --

def test_triggers_at_configured_iteration_interval():
    fake_llm = FakeLLM(response="SUMMARY-A")
    agent = _make_agent(summarize_every=5, summarizer_llm=fake_llm)
    agent.scratchpad = "x" * 100

    assert not agent._summarize_should_trigger(3)  # not a multiple of 5
    assert agent._summarize_should_trigger(5)
    agent._do_summarize(5)
    assert "SUMMARY-A" in agent.scratchpad
    assert fake_llm.calls == 1


def test_triggers_when_over_char_limit():
    fake_llm = FakeLLM(response="SUMMARY-B")
    # summarize_every huge -- never a multiple within any real test iteration
    # count -- isolates the char-threshold trigger path.
    agent = _make_agent(summarize_every=1000, max_scratchpad_chars=50, summarizer_llm=fake_llm)
    agent.scratchpad = "y" * 100

    assert agent._summarize_should_trigger(1)
    agent._do_summarize(1)
    assert "SUMMARY-B" in agent.scratchpad
    assert fake_llm.calls == 1


def test_char_threshold_does_not_retrigger_when_summary_stays_over_limit():
    """Regression: the char-threshold check compared scratchpad length
    against max_scratchpad_chars every call, including AFTER a
    summarization already ran. If the summary itself is still longer than
    max_scratchpad_chars (small threshold, or a verbose summarizer LLM),
    every subsequent check re-triggered summarization on the
    already-compressed text -- one LLM call per iteration forever, even
    with no new content to compress. It must only re-trigger once the
    scratchpad actually grows past where it was right after the last
    summarization."""
    long_summary = "S" * 80  # deliberately still over max_scratchpad_chars below
    fake_llm = FakeLLM(response=long_summary)
    agent = _make_agent(summarize_every=1000, max_scratchpad_chars=50, summarizer_llm=fake_llm)
    agent.scratchpad = "x" * 100

    assert agent._summarize_should_trigger(1)
    agent._do_summarize(1)
    assert fake_llm.calls == 1
    length_after_first = len(agent.scratchpad)
    assert length_after_first > 50  # summary is still over threshold

    # Scratchpad unchanged since the last summarization -- must NOT retrigger.
    assert not agent._summarize_should_trigger(2)
    assert not agent._summarize_should_trigger(3)
    assert fake_llm.calls == 1

    # New content appended (scratchpad grew past the post-summary watermark)
    # -- this legitimately should retrigger.
    agent.scratchpad += "new observation " * 5
    assert agent._summarize_should_trigger(4)
    agent._do_summarize(4)
    assert fake_llm.calls == 2


def test_concurrent_summarize_attempt_for_same_agent_skips_without_blocking():
    """The non-blocking lock still applies -- a concurrent _do_summarize()
    call for the SAME agent skips immediately rather than waiting."""
    fake_llm = FakeLLM(response="SUMMARIZED")
    agent = _make_agent(summarize_every=1, summarizer_llm=fake_llm)
    agent.scratchpad = "z" * 100

    agent._summarizer_lock.acquire()  # simulate an in-progress summarization
    try:
        agent._do_summarize(1)  # must skip immediately, not block
    finally:
        agent._summarizer_lock.release()

    assert fake_llm.calls == 0
    assert agent.scratchpad == "z" * 100


def test_empty_summary_logs_warning_and_leaves_scratchpad_unchanged(caplog):
    """An LLM returning an empty/whitespace-only summary must leave the
    scratchpad untouched, and log a warning naming the iteration."""
    fake_llm = FakeLLM(response="   ")  # whitespace-only -> strips to empty
    agent = _make_agent(summarize_every=1, summarizer_llm=fake_llm)
    agent.scratchpad = "x" * 100

    with caplog.at_level(logging.WARNING, logger="autourgos_agent"):
        agent._do_summarize(1)

    assert agent.scratchpad == "x" * 100  # unchanged
    assert any("empty summary" in r.message for r in caplog.records)


def test_narrates_via_agent_logger_middleware_on_success():
    """On genuine compression success, must call agent.logger.middleware(...)
    with source 'Summarizer' and a message describing the compression."""
    fake_llm = FakeLLM(response="SUMMARY-NARRATE")
    agent = _make_agent(summarize_every=1, summarizer_llm=fake_llm)
    agent.scratchpad = "w" * 100
    agent.logger = MagicMock()  # spy, replacing the real AgentLogger

    agent._do_summarize(1)

    assert agent.logger.middleware.called
    args, _ = agent.logger.middleware.call_args
    assert args[0] == "Summarizer"
    assert "Compressed scratchpad" in args[1]
    assert "iteration 1" in args[1]


def test_native_mode_skips_summarization_and_warns_once(caplog):
    """
    Regression: tool_calling_mode="native" never sends agent.scratchpad to
    the LLM (it's a human-readable trace only -- the real conversation
    state is an internal message list this has no access to). Summarizing
    it there used to silently burn a real LLM call compressing text the
    model never sees. It must now skip summarization for a native-mode
    agent and warn once (not per iteration).
    """
    fake_llm = FakeLLM(response="SUMMARY-SHOULD-NOT-APPEAR")
    agent = _make_agent(summarize_every=1, summarizer_llm=fake_llm, tool_calling_mode="native")
    agent.scratchpad = "x" * 100

    with caplog.at_level("WARNING"):
        for iteration in (1, 2, 3):
            agent._maybe_summarize(iteration)

    assert fake_llm.calls == 0
    assert agent.scratchpad == "x" * 100
    native_mode_warnings = [r for r in caplog.records if "native" in r.message]
    assert len(native_mode_warnings) == 1


# -- end-to-end via Agent(summarize_every=..., summarizer_llm=..., ...) ------

def test_agent_without_summarize_every_never_summarizes():
    """summarize_every=None (default) -- fully disabled, matching prior
    behavior. Only the plain char-trim applies, never LLM summarization."""
    responses = [
        json.dumps({
            "thought": "gathering info",
            "actions": [{"action": "echo", "action_input": {"text": "x" * 50}}],
            "final_answer": None,
        }),
        _final_response("done"),
    ]
    agent = make_test_agent(responses=responses)
    result = agent.invoke("do something")

    assert result == "done"
    assert "[Summarized" not in agent.scratchpad


def test_agent_summarize_every_kwarg_uses_shared_max_scratchpad_chars():
    llm = ScriptedFakeLLM([_final_response("done")])
    agent = Agent(llm=llm, summarize_every=3, max_scratchpad_chars=1234)

    assert agent.summarize_every == 3
    # The trim cap (MAX_SCRATCHPAD_CHARS) and the summarizer's own threshold
    # share this one value, by design.
    assert agent.MAX_SCRATCHPAD_CHARS == 1234


def test_agent_summarize_every_kwarg_actually_summarizes_using_agents_own_llm():
    """End-to-end: no summarizer_llm= passed, so it must fall back to this
    agent's own llm -- and the resulting summary must actually land on
    agent.scratchpad, not just get computed and discarded."""
    llm = ScriptedFakeLLM([
        json.dumps({
            "thought": "gathering info",
            "actions": [{"action": "echo", "action_input": {"text": "x" * 50}}],
            "final_answer": None,
        }),
        "SUMMARY-BUILTIN-KWARG",
        _final_response("done"),
    ])
    agent = Agent(llm=llm, summarize_every=1, max_iterations=10)
    agent.add_tools(_ECHO_TOOL)

    result = agent.invoke("do something")

    assert result == "done"
    assert "SUMMARY-BUILTIN-KWARG" in agent.scratchpad


def test_agent_summarizer_llm_kwarg_used_instead_of_agents_own_llm():
    """summarizer_llm=, when given, must be the LLM actually used for
    summarization -- the main loop's own llm must never see the
    summarization prompt, and vice versa."""
    main_llm = ScriptedFakeLLM([
        json.dumps({
            "thought": "gathering info",
            "actions": [{"action": "echo", "action_input": {"text": "x" * 50}}],
            "final_answer": None,
        }),
        _final_response("done"),
    ])
    summarizer_llm = ScriptedFakeLLM(["SUMMARY-FROM-DEDICATED-LLM"])
    agent = Agent(llm=main_llm, summarize_every=1, summarizer_llm=summarizer_llm, max_iterations=10)
    agent.add_tools(_ECHO_TOOL)

    result = agent.invoke("do something")

    assert result == "done"
    assert "SUMMARY-FROM-DEDICATED-LLM" in agent.scratchpad
    assert summarizer_llm.call_count == 1
    # main_llm's 2 canned responses were both consumed by the main loop
    # (tool-call turn + final-answer turn) -- none of them went to the
    # summarizer.
    assert main_llm.call_count == 2


def test_agent_summarize_every_current_query_is_used_in_summarize_prompt():
    """agent.current_query must be used to fill the {query} slot in the
    summarization prompt."""
    class _RecordingLLM:
        def __init__(self) -> None:
            self.prompts = []

        def invoke(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "[compressed]"

    main_llm = ScriptedFakeLLM([
        json.dumps({
            "thought": "gathering info",
            "actions": [{"action": "echo", "action_input": {"text": "y" * 200}}],
            "final_answer": None,
        }),
        _final_response("done"),
    ])
    summarizer_llm = _RecordingLLM()
    agent = Agent(
        llm=main_llm, summarize_every=1, max_scratchpad_chars=20,
        summarizer_llm=summarizer_llm, max_iterations=10,
    )
    agent.add_tools(_ECHO_TOOL)

    agent.invoke("what is the real query text")

    assert len(summarizer_llm.prompts) == 1
    assert "what is the real query text" in summarizer_llm.prompts[0]


def test_multiple_agents_with_own_summarize_every_dont_interfere():
    """Two independent Agent instances, each with their own summarize_every,
    must not share any state (no more cross-instance registry -- this is
    now plain per-instance state)."""
    llm_a = FakeLLM(response="SUMMARY-AGENT-A")
    llm_b = FakeLLM(response="SUMMARY-AGENT-B")
    agent_a = _make_agent(summarize_every=1, summarizer_llm=llm_a)
    agent_b = _make_agent(summarize_every=1, summarizer_llm=llm_b)
    agent_a.scratchpad = "a" * 100
    agent_b.scratchpad = "b" * 100

    agent_a._do_summarize(1)
    agent_b._do_summarize(1)

    assert "SUMMARY-AGENT-A" in agent_a.scratchpad
    assert "SUMMARY-AGENT-B" in agent_b.scratchpad
    assert llm_a.calls == 1
    assert llm_b.calls == 1
