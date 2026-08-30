"""
Regression tests for the Phase 0 stabilization fixes (see idea.md at the
workspace root):

  A1 -- async tools silently no-op in sync invoke()
  A2 -- native mode never re-reads agent.system_prompt after loop start
  B1 -- native mode's messages list has no context-window budget
  B4 -- invoke()/ainvoke() require at least one tool
  B5 -- MAX_SCRATCHPAD_CHARS / MAX_TOOL_OUTPUT_CHARS / MAX_TOOL_WORKERS
        were class-attribute-only, not constructor parameters
"""

from __future__ import annotations

import asyncio
import json

from autourgos_agent import Agent, CallbackHandler, tool
from autourgos_agent.testing import ScriptedFakeLLM, ScriptedToolCallLLM


def _final(text: str) -> str:
    return json.dumps({"thought": None, "actions": [], "final_answer": text})


def _tool_action(name: str, input_: dict) -> str:
    return json.dumps(
        {"thought": "t", "actions": [{"action": name, "action_input": input_}], "final_answer": None}
    )


# ── A1 ──────────────────────────────────────────────────────────────────────

def test_async_tool_actually_runs_under_sync_invoke():
    calls = {"ran": False}

    @tool
    async def fetch(url: str) -> str:
        """Fetch a url."""
        await asyncio.sleep(0)
        calls["ran"] = True
        return "REAL_RESULT"

    llm = ScriptedFakeLLM([
        _tool_action("fetch", {"url": "x"}),
        _final("done"),
    ])
    agent = Agent(llm=llm)
    agent.add_tools(fetch)

    result = agent.invoke("go")

    assert calls["ran"] is True
    assert result == "done"
    assert "REAL_RESULT" in agent.scratchpad
    assert "coroutine object" not in agent.scratchpad


def test_async_tool_error_still_reported_under_sync_invoke():
    @tool
    async def boom() -> str:
        """Always fails."""
        await asyncio.sleep(0)
        raise RuntimeError("kaboom")

    llm = ScriptedFakeLLM([
        _tool_action("boom", {}),
        _final("done"),
    ])
    agent = Agent(llm=llm)
    agent.add_tools(boom)

    agent.invoke("go")

    assert "kaboom" in agent.scratchpad


# ── B4 ──────────────────────────────────────────────────────────────────────

def test_invoke_allows_empty_tool_list_prompt_mode():
    llm = ScriptedFakeLLM([_final("no tools needed")])
    agent = Agent(llm=llm)

    result = agent.invoke("just answer directly")

    assert result == "no tools needed"


def test_invoke_allows_empty_tool_list_native_mode():
    llm = ScriptedToolCallLLM([ScriptedToolCallLLM.final("no tools needed")])
    agent = Agent(llm=llm, tool_calling_mode="native")

    result = agent.invoke("just answer directly")

    assert result == "no tools needed"


# ── B5 ──────────────────────────────────────────────────────────────────────

def test_per_instance_output_limits_override_class_defaults():
    llm = ScriptedFakeLLM([_final("ok")])
    agent = Agent(
        llm=llm,
        max_scratchpad_chars=123,
        max_tool_output_chars=45,
        max_tool_workers=2,
    )

    assert agent.MAX_SCRATCHPAD_CHARS == 123
    assert agent.MAX_TOOL_OUTPUT_CHARS == 45
    assert agent.MAX_TOOL_WORKERS == 2

    # unrelated instance is untouched -- confirms this is a per-instance
    # override, not an accidental mutation of the shared class attribute
    other = Agent(llm=llm)
    assert other.MAX_SCRATCHPAD_CHARS == 15_000
    assert other.MAX_TOOL_OUTPUT_CHARS == 5_000
    assert other.MAX_TOOL_WORKERS == 8


def test_max_tool_output_chars_override_actually_truncates():
    @tool
    def echo() -> str:
        """Return a long string."""
        return "x" * 1000

    llm = ScriptedFakeLLM([
        _tool_action("echo", {}),
        _final("done"),
    ])
    agent = Agent(llm=llm, max_tool_output_chars=10)
    agent.add_tools(echo)

    agent.invoke("go")

    assert "[truncated]" in agent.scratchpad
    # the raw 1000-char payload should not appear whole
    assert "x" * 1000 not in agent.scratchpad


# ── A2 + B1 ───────────────────────────────────────────────────────────────

class _RecordingLLM(ScriptedToolCallLLM):
    """Records the system-role message contents seen on every
    invoke_with_tools() call, so a test can assert what the model actually
    received each iteration -- not just what agent.system_prompt held at
    the end of the run."""

    def __init__(self, responses):
        super().__init__(responses)
        self.seen_system_messages: list[list[str]] = []
        self.seen_message_char_sizes: list[int] = []

    def invoke_with_tools(self, prompt, tools, **kwargs):
        self.seen_system_messages.append(
            [m.get("content") for m in prompt if m.get("role") == "system"]
        )
        self.seen_message_char_sizes.append(len(json.dumps(prompt, default=str)))
        return super().invoke_with_tools(prompt, tools, **kwargs)


def test_native_mode_sees_live_system_prompt_edits_mid_run():
    @tool
    def ping(x: str) -> str:
        """Ping."""
        return "pong"

    class Steerer(CallbackHandler):
        def on_iteration_start(self, iteration, agent=None, **kwargs):
            if iteration == 2 and agent is not None:
                agent.system_prompt += "\n\n[OVERRIDE] stop and answer now."

    llm = _RecordingLLM([
        ScriptedToolCallLLM.tool_call("ping", {"x": "a"}, "c1"),
        ScriptedToolCallLLM.tool_call("ping", {"x": "b"}, "c2"),
        ScriptedToolCallLLM.final("done"),
    ])
    agent = Agent(
        llm=llm,
        tool_calling_mode="native",
        system_prompt="BASE",
        middleware=[Steerer()],
    )
    agent.add_tools(ping)

    agent.invoke("go")

    # on_iteration_start fires before that same iteration's LLM call, so
    # the override (injected at the start of iteration 2) is already
    # visible in iteration 2's own call -- iteration 1 is unaffected.
    assert llm.seen_system_messages[0] == ["BASE"]
    assert llm.seen_system_messages[1] == ["BASE\n\n[OVERRIDE] stop and answer now."]
    assert llm.seen_system_messages[2] == ["BASE\n\n[OVERRIDE] stop and answer now."]


def test_native_mode_messages_list_is_trimmed_to_budget():
    @tool
    def big() -> str:
        """Return a big chunk of text."""
        return "y" * 500

    responses = [ScriptedToolCallLLM.tool_call("big", {}, f"c{i}") for i in range(6)]
    responses.append(ScriptedToolCallLLM.final("done"))
    llm = _RecordingLLM(responses)
    # small enough that a handful of 500-char tool outputs will overflow it
    agent = Agent(llm=llm, tool_calling_mode="native", max_scratchpad_chars=800)
    agent.add_tools(big)

    result = agent.invoke("go")

    assert result == "done"
    # every call the LLM actually received stayed near the configured cap --
    # without trimming, the last call's payload would include all 6 prior
    # 500-char tool turns (~3000+ chars) plus the running total.
    assert max(llm.seen_message_char_sizes) < 1500
    # and the cap was genuinely exercised, not just trivially satisfied
    assert llm.seen_message_char_sizes[-1] > 800 - 200  # still has at least one turn's worth
