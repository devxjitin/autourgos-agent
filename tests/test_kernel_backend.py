"""
Tests for Agent(backend="kernel") -- the opt-in bridge to autourgos-kernel
(Phase 2, Part B of the Autourgos v3 roadmap; see idea.md and
autourgos_agent/kernel_backend.py's module docstring for the documented
gaps vs backend="legacy").

These require autourgos-kernel to be installed (pip install
autourgos-agent[kernel]) -- skipped entirely otherwise, since
backend="kernel" is optional and must not break `pip install
autourgos-agent` on its own.

backend="legacy" (the default, tested everywhere else in this suite) is
untouched by any of this -- these tests exist to verify the *new*, opt-in
path, not to duplicate the existing 95 tests.
"""

from __future__ import annotations

import json

import pytest
from autourgos_core import Action, Risk
from autourgos_policy import PolicyConfig, PolicyExecutor, PolicyGate

from autourgos_agent import Agent, AgentMaxIterationsError, CallbackHandler, tool
from autourgos_agent.testing import ScriptedFakeLLM, ScriptedToolCallLLM

autourgos_kernel = pytest.importorskip("autourgos_kernel")


def _final(text: str) -> str:
    return json.dumps({"thought": None, "actions": [], "final_answer": text})


def _action(name: str, input_: dict) -> str:
    return json.dumps(
        {"thought": "t", "actions": [{"action": name, "action_input": input_}], "final_answer": None}
    )


def test_kernel_backend_rejects_unknown_backend_value():
    llm = ScriptedFakeLLM([_final("x")])
    with pytest.raises(ValueError):
        Agent(llm=llm, backend="not-a-real-backend")


def test_policy_options_are_kernel_only_and_effect_budget_is_validated():
    llm = ScriptedFakeLLM([_final("x")])
    with pytest.raises(ValueError, match="backend='kernel'"):
        Agent(llm=llm, capabilities=[])
    with pytest.raises(ValueError, match="max_effects"):
        Agent(llm=llm, backend="kernel", max_effects=-1)


def test_kernel_backend_final_answer_no_tools():
    llm = ScriptedFakeLLM([_final("hello world")])
    agent = Agent(llm=llm, backend="kernel")

    result = agent.invoke("just answer directly")

    assert result == "hello world"


def test_kernel_backend_prompt_mode_tool_call_then_final_answer():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    llm = ScriptedFakeLLM([_action("add", {"a": 2, "b": 3}), _final("The sum is 5")])
    agent = Agent(llm=llm, backend="kernel")
    agent.add_tools(add)

    result = agent.invoke("add 2 and 3")

    assert result == "The sum is 5"
    assert "5" in agent.scratchpad


def test_kernel_backend_native_mode_tool_call_then_final_answer():
    @tool
    def ping(x: str) -> str:
        """Ping."""
        return f"pong:{x}"

    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("ping", {"x": "a"}, "c1"),
        ScriptedToolCallLLM.final("done"),
    ])
    agent = Agent(llm=llm, backend="kernel")
    agent.add_tools(ping)

    result = agent.invoke("go")

    assert result == "done"
    assert "pong:a" in agent.scratchpad


def test_kernel_backend_fires_the_same_callback_hooks_as_legacy():
    @tool
    def ping(x: str) -> str:
        """Ping."""
        return f"pong:{x}"

    events = []

    class Spy(CallbackHandler):
        def on_agent_start(self, query, agent=None, **kw):
            events.append(("agent_start", query))

        def on_agent_end(self, result, agent=None, **kw):
            events.append(("agent_end", result))

        def on_tool_start(self, tool_name, tool_input, agent=None, **kw):
            events.append(("tool_start", tool_name, tool_input))

        def on_tool_end(self, tool_name, result, agent=None, **kw):
            events.append(("tool_end", tool_name, result))

    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("ping", {"x": "a"}, "c1"),
        ScriptedToolCallLLM.final("done"),
    ])
    agent = Agent(llm=llm, backend="kernel", middleware=[Spy()])
    agent.add_tools(ping)

    agent.invoke("go")

    assert events == [
        ("agent_start", "go"),
        ("tool_start", "ping", {"x": "a"}),
        ("tool_end", "ping", "pong:a"),
        ("agent_end", "done"),
    ]


def test_kernel_backend_max_iterations_raises_agent_error_not_kernel_error():
    llm = ScriptedFakeLLM([_action("noop", {})] * 5)
    agent = Agent(llm=llm, backend="kernel", max_iterations=2)
    agent.add_tools({
        "name": "noop",
        "description": "noop",
        "parameters": {"type": "object", "properties": {}},
        "func": lambda: "ok",
    })

    with pytest.raises(AgentMaxIterationsError):
        agent.invoke("never finishes")


def test_kernel_backend_system_prompt_reaches_the_model():
    class RecordingLLM(ScriptedToolCallLLM):
        def __init__(self, responses):
            super().__init__(responses)
            self.seen_system_messages = []

        async def ainvoke_with_tools(self, messages, tools, **kwargs):
            self.seen_system_messages.append(
                [m.get("content") for m in messages if m.get("role") == "system"]
            )
            return await super().ainvoke_with_tools(messages, tools, **kwargs)

    llm = RecordingLLM([ScriptedToolCallLLM.final("done")])
    agent = Agent(llm=llm, backend="kernel", system_prompt="BASE")

    agent.invoke("go")

    assert llm.seen_system_messages == [["BASE"]]


def test_kernel_backend_memory_integration():
    class SimpleMemory:
        def __init__(self):
            self.history = []

        def add_user_message(self, m):
            self.history.append(("user", m))

        def add_agent_message(self, m):
            self.history.append(("agent", m))

        def get_history(self):
            return [{"role": r, "content": c} for r, c in self.history]

    mem = SimpleMemory()
    llm = ScriptedFakeLLM([_final("first answer"), _final("second answer")])
    agent = Agent(llm=llm, backend="kernel", memory=mem)

    r1 = agent.invoke("question one")
    r2 = agent.invoke("question two")

    assert r1 == "first answer"
    assert r2 == "second answer"
    assert mem.history == [
        ("user", "question one"),
        ("agent", "first answer"),
        ("user", "question two"),
        ("agent", "second answer"),
    ]


@pytest.mark.asyncio
async def test_kernel_backend_ainvoke_works():
    llm = ScriptedFakeLLM([_final("async result")])
    agent = Agent(llm=llm, backend="kernel")

    result = await agent.ainvoke("go")

    assert result == "async result"


def test_kernel_backend_approval_callback_can_deny_a_tool_call():
    @tool
    def dangerous() -> str:
        """Do something dangerous."""
        return "did it"

    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("dangerous", {}, "c1"),
        ScriptedToolCallLLM.final("done"),
    ])
    agent = Agent(llm=llm, backend="kernel", approval_callback=lambda name, args: False)
    agent.add_tools(dangerous)

    agent.invoke("go")

    assert "denied" in agent.scratchpad


def _trusted_policy(_run):
    return PolicyExecutor(
        PolicyGate(PolicyConfig(profile="trusted", require_targets=False))
    )


def test_kernel_policy_factory_executes_described_tool_and_is_per_run():
    calls = []
    factory_run_ids = []

    def describe(call):
        return Action(tool=call.name, arguments=call.arguments, risk=Risk.READ)

    @tool(describe=describe, capability="records", risk="read")
    def record(value: str) -> str:
        calls.append(value)
        return f"recorded:{value}"

    def factory(run):
        factory_run_ids.append(run.run_id)
        return _trusted_policy(run)

    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("record", {"value": "a"}, "c1"),
        ScriptedToolCallLLM.final("one"),
        ScriptedToolCallLLM.tool_call("record", {"value": "b"}, "c2"),
        ScriptedToolCallLLM.final("two"),
    ])
    agent = Agent(
        llm=llm,
        backend="kernel",
        policy_executor_factory=factory,
    )
    agent.add_tools(record)

    assert agent.invoke("first") == "one"
    assert agent.invoke("second") == "two"
    assert calls == ["a", "b"]
    assert len(set(factory_run_ids)) == 2


def test_kernel_policy_preserves_legacy_approval_exactly_once():
    approvals = []
    calls = []

    def describe(call):
        return Action(tool=call.name, arguments=call.arguments, risk=Risk.READ)

    @tool(describe=describe)
    def record(value: str) -> str:
        calls.append(value)
        return "ok"

    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("record", {"value": "a"}, "c1"),
        ScriptedToolCallLLM.final("done"),
    ])
    agent = Agent(
        llm=llm,
        backend="kernel",
        policy_executor_factory=_trusted_policy,
        approval_callback=lambda name, args: approvals.append((name, args)) or True,
    )
    agent.add_tools(record)

    agent.invoke("go")

    assert approvals == [("record", {"value": "a"})]
    assert calls == ["a"]


def test_kernel_max_effects_reaches_concurrent_policy_boundary():
    calls = []

    def describe(call):
        return Action(tool=call.name, arguments=call.arguments, risk=Risk.READ)

    @tool(describe=describe)
    def record(value: str) -> str:
        calls.append(value)
        return value

    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.calls_([
            {"name": "record", "arguments": {"value": "a"}, "call_id": "a"},
            {"name": "record", "arguments": {"value": "b"}, "call_id": "b"},
        ]),
        ScriptedToolCallLLM.final("done"),
    ])
    agent = Agent(
        llm=llm,
        backend="kernel",
        policy_executor_factory=_trusted_policy,
        max_effects=1,
    )
    agent.add_tools(record)

    agent.invoke("go")

    assert len(calls) == 1
    assert "Effect budget exhausted" in agent.scratchpad
