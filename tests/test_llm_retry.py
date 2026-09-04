"""
Regression tests for Agent's llm_retries/llm_retry_backoff/llm_retry_on --
_call_llm_with_retry()/_acall_llm_with_retry() now delegate to
autourgos_core.retry_with_backoff()/aretry_with_backoff() (Sprint 4b).
These tests exist to prove the migration is behavior-preserving; no
dedicated retry test file existed before this.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from autourgos_agent import Agent, AgentLLMError
from autourgos_agent.base import BaseLLM


def _final(text: str) -> str:
    return json.dumps({"thought": None, "actions": [], "final_answer": text})


class FlakyLLM(BaseLLM):
    """Fails `fail_times` times (raising RuntimeError), then returns a
    final-answer response. No network calls."""

    def __init__(self, fail_times: int, exc_factory=lambda: RuntimeError("transient")) -> None:
        self.fail_times = fail_times
        self.exc_factory = exc_factory
        self.calls = 0

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc_factory()
        return _final("recovered")

    async def ainvoke(self, prompt, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc_factory()
        return _final("recovered")


def test_llm_retries_zero_is_single_call_no_retry():
    """Default llm_retries=0 -- a failure surfaces immediately, matching
    prior behavior (no retry loop overhead when not opted in)."""
    llm = FlakyLLM(fail_times=1)
    agent = Agent(llm=llm)
    with pytest.raises(AgentLLMError):
        agent.invoke("hi")
    assert llm.calls == 1


def test_llm_retries_recovers_after_transient_failures():
    llm = FlakyLLM(fail_times=2)
    agent = Agent(llm=llm, llm_retries=3, llm_retry_backoff=0.001)
    result = agent.invoke("hi")
    assert result == "recovered"
    assert llm.calls == 3


def test_llm_retries_exhausted_raises():
    llm = FlakyLLM(fail_times=5)
    agent = Agent(llm=llm, llm_retries=2, llm_retry_backoff=0.001)
    with pytest.raises(AgentLLMError):
        agent.invoke("hi")
    assert llm.calls == 3  # 1 initial + 2 retries


def test_llm_retry_on_custom_predicate_stops_retry():
    """A custom llm_retry_on that returns False must stop retrying
    immediately, even with llm_retries > 0."""
    llm = FlakyLLM(fail_times=5, exc_factory=lambda: ValueError("non-retryable"))
    agent = Agent(llm=llm, llm_retries=3, llm_retry_backoff=0.001, llm_retry_on=lambda exc: False)
    with pytest.raises(AgentLLMError):
        agent.invoke("hi")
    assert llm.calls == 1


def test_llm_retry_max_backoff_caps_delay(monkeypatch):
    sleeps = []
    monkeypatch.setattr("autourgos_core.concurrency.time.sleep", lambda s: sleeps.append(s))

    llm = FlakyLLM(fail_times=4)
    agent = Agent(llm=llm, llm_retries=4, llm_retry_backoff=1.0, llm_retry_max_backoff=2.0)
    agent.invoke("hi")

    # 2**0, 2**1, 2**2, 2**3 = 1, 2, 4, 8 -- capped at 2.0 from the 2nd retry onward
    assert sleeps == [1.0, 2.0, 2.0, 2.0]


def test_async_llm_retries_recovers_after_transient_failures():
    llm = FlakyLLM(fail_times=2)
    agent = Agent(llm=llm, llm_retries=3, llm_retry_backoff=0.001)

    async def run():
        return await agent.ainvoke("hi")

    result = asyncio.run(run())
    assert result == "recovered"
    assert llm.calls == 3
