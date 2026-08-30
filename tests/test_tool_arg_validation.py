"""
Tests for tool-argument validation in BaseAgent._execute_tool /
_execute_tool_async.

Regression coverage: previously a wrong/missing tool argument from the
model went straight into `func(**tool_input)`, so the model saw a raw
Python TypeError (e.g. "greet() missing 1 required positional argument:
'name'") as the tool's "Observation" -- confusing, since it names Python
internals instead of pointing at the tool's schema. Args are now bound
against the tool's signature before the call, so a mismatch returns a
clear message naming the expected signature instead of calling the tool
at all.
"""
from __future__ import annotations

import json

import pytest

from autourgos_agent import Agent
from autourgos_agent.testing import ScriptedFakeLLM


def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"


GREET_TOOL = {
    "name": "greet",
    "description": "Greets someone",
    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    "func": greet,
}


def _agent_with_one_tool_call(tool_input):
    responses = [
        json.dumps({
            "thought": "call greet",
            "actions": [{"action": "greet", "action_input": tool_input}],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
    ]
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=5)
    agent.add_tools(GREET_TOOL)
    return agent


def test_missing_required_argument_gives_clear_message_not_raw_typeerror():
    agent = _agent_with_one_tool_call({})  # missing required 'name'
    agent.invoke("greet someone")

    assert "greet" in agent.scratchpad
    assert "invalid arguments" in agent.scratchpad
    assert "Expected signature" in agent.scratchpad


def test_unexpected_keyword_argument_gives_clear_message():
    agent = _agent_with_one_tool_call({"name": "Ana", "age": 30})  # 'age' not a param
    agent.invoke("greet someone")

    assert "invalid arguments" in agent.scratchpad
    assert "Expected signature" in agent.scratchpad
    assert "Expected signature: greet(" in agent.scratchpad
    assert "greeting" in agent.scratchpad


def test_valid_arguments_still_call_the_tool_normally():
    agent = _agent_with_one_tool_call({"name": "Ana"})
    result = agent.invoke("greet someone")

    assert result == "done"
    assert "Hello, Ana!" in agent.scratchpad
    assert "invalid arguments" not in agent.scratchpad


def test_runtime_error_inside_tool_body_is_unaffected_by_validation():
    """A tool that raises for reasons unrelated to argument binding (e.g. a
    bad value that passes the signature but fails inside the function) must
    still surface as the existing generic execution error, not be swallowed
    or misreported as an argument-schema problem."""

    def divide(a: int, b: int) -> float:
        return a / b

    tool = {
        "name": "divide",
        "description": "Divides a by b",
        "parameters": {"type": "object", "properties": {}},
        "func": divide,
    }
    responses = [
        json.dumps({
            "thought": "divide",
            "actions": [{"action": "divide", "action_input": {"a": 1, "b": 0}}],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
    ]
    agent = Agent(llm=ScriptedFakeLLM(responses), max_iterations=5)
    agent.add_tools(tool)
    agent.invoke("divide")

    assert "Error executing 'divide'" in agent.scratchpad
    assert "invalid arguments" not in agent.scratchpad


@pytest.mark.asyncio
async def test_async_path_also_validates_arguments():
    agent = _agent_with_one_tool_call({})
    await agent.ainvoke("greet someone")

    assert "invalid arguments" in agent.scratchpad
    assert "Expected signature" in agent.scratchpad
