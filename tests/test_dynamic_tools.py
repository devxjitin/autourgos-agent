"""
Regression tests for tool_map/tool-list staleness: a tool added mid-run
(e.g. by a middleware calling agent.add_tools() from inside a tool call,
the way autourgos-toolbox's expose_toolbox() does) must be both advertised
to the LLM and actually callable on the very next iteration -- not just
present on agent.tools while the loop's own tool_map stays frozen from
before the run started.
"""

from __future__ import annotations

import json

from autourgos_agent import Agent, CallbackHandler
from autourgos_agent.testing import make_test_agent, ScriptedToolCallLLM


def _final(text: str) -> str:
    return json.dumps({"thought": None, "actions": [], "final_answer": text})


def _late_tool_action(name: str, input_: dict) -> str:
    return json.dumps({"thought": "calling late tool", "actions": [{"action": name, "action_input": input_}], "final_answer": None})


def test_tool_added_mid_run_is_actually_callable_prompt_mode():
    added = {"done": False}

    def add_late_tool() -> str:
        return "added"

    class AddsToolMiddleware(CallbackHandler):
        def on_tool_end(self, tool_name, result, agent=None, **kwargs):
            if tool_name == "trigger" and not added["done"]:
                agent.add_tools({
                    "name": "late_tool",
                    "description": "Only available after trigger runs.",
                    "parameters": {"type": "object", "properties": {}},
                    "func": lambda: "late tool result",
                })
                added["done"] = True

    responses = [
        json.dumps({"thought": None, "actions": [{"action": "trigger", "action_input": {}}], "final_answer": None}),
        _late_tool_action("late_tool", {}),
        _final("done"),
    ]
    agent = make_test_agent(
        responses=responses,
        tools=[{
            "name": "trigger", "description": "trigger", "parameters": {"type": "object", "properties": {}},
            "func": lambda: "triggered",
        }],
        middleware=[AddsToolMiddleware()],
        max_iterations=5,
    )
    result = agent.invoke("go")

    assert result == "done"
    assert "late tool result" in agent.scratchpad
    assert "not found" not in agent.scratchpad
    # the prompt for iteration 2 must have advertised late_tool too
    assert "late_tool" in str(agent.llm.calls[1]["prompt"])


def test_tool_added_mid_run_is_actually_callable_native_mode():
    added = {"done": False}

    class AddsToolMiddleware(CallbackHandler):
        def on_tool_end(self, tool_name, result, agent=None, **kwargs):
            if tool_name == "trigger" and not added["done"]:
                agent.add_tools({
                    "name": "late_tool",
                    "description": "Only available after trigger runs.",
                    "parameters": {"type": "object", "properties": {}},
                    "func": lambda: "late tool result",
                })
                added["done"] = True

    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("trigger", {}, call_id="c1"),
        ScriptedToolCallLLM.tool_call("late_tool", {}, call_id="c2"),
        ScriptedToolCallLLM.final("done"),
    ])
    agent = Agent(llm=llm, tool_calling_mode="native", middleware=[AddsToolMiddleware()], max_iterations=5)
    agent.add_tools({
        "name": "trigger", "description": "trigger", "parameters": {"type": "object", "properties": {}},
        "func": lambda: "triggered",
    })

    result = agent.invoke("go")

    assert result == "done"
    assert "late tool result" in agent.scratchpad
    assert "not found" not in agent.scratchpad
