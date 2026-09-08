"""
Coverage for autourgos_agent._toolbox._ToolboxRuntime's collision guard
and expose/restore lifecycle -- previously only incidentally exercised
(autourgos-audit-report.md). Built by Agent(toolbox=[...]) internally,
not middleware -- see _toolbox.py's module docstring.
"""
import json

import pytest

from autourgos_core import Toolbox
from autourgos_agent._toolbox import _ToolboxRuntime
from autourgos_agent.testing import make_test_agent


def _tool(name, result="ok"):
    return {
        "name": name,
        "description": name,
        "parameters": {"type": "object", "properties": {}},
        "func": lambda: result,
    }


def _final(text: str) -> str:
    return json.dumps({"thought": None, "actions": [], "final_answer": text})


class DummyAgent:
    def __init__(self, tools=None):
        self.tools = list(tools or [])
        self.logger = None

    def add_tools(self, *tools):
        # Mirrors the real Agent.add_tools(): a same-named tool replaces
        # (not duplicates) the existing one in the list.
        for tool in tools:
            name = tool["name"] if isinstance(tool, dict) else tool.name
            self.tools = [t for t in self.tools if (t["name"] if isinstance(t, dict) else t.name) != name]
            self.tools.append(tool)


# ── collision guard ──────────────────────────────────────────────────────

def test_add_toolbox_rejects_cross_toolbox_name_collision():
    runtime = _ToolboxRuntime()
    runtime.add_toolbox("a", "toolbox a", [_tool("shared")])

    with pytest.raises(ValueError, match="shared"):
        runtime.add_toolbox("b", "toolbox b", [_tool("shared")])


def test_add_toolbox_allows_reregistering_same_name():
    """Re-adding a toolbox under its own name (e.g. updating its tool list)
    must not self-collide against its own prior registration."""
    runtime = _ToolboxRuntime()
    runtime.add_toolbox("a", "toolbox a", [_tool("x")])
    runtime.add_toolbox("a", "toolbox a v2", [_tool("x"), _tool("y")])

    assert [t["name"] for t in runtime.toolboxes["a"].tools] == ["x", "y"]


def test_start_rejects_reserved_tool_name_collision():
    runtime = _ToolboxRuntime(toolboxes=[Toolbox("a", "toolbox a", [_tool("x")])])
    agent = DummyAgent(tools=[_tool("expose_toolbox")])

    with pytest.raises(ValueError, match="expose_toolbox"):
        runtime.start(agent)


# ── exposure / restore lifecycle, end-to-end via a real agent loop ──────

def test_expose_toolbox_makes_tools_callable_and_restores_after_run():
    responses = [
        json.dumps({"thought": "t", "actions": [{"action": "expose_toolbox", "action_input": {"toolbox_name": "math"}}], "final_answer": None}),
        json.dumps({"thought": "t", "actions": [{"action": "add_two", "action_input": {}}], "final_answer": None}),
        _final("done"),
    ]

    agent = make_test_agent(
        responses=responses,
        toolbox=[Toolbox("math", "math tools", [_tool("add_two", result=4)])],
        max_iterations=5,
    )
    result = agent.invoke("go")

    assert result == "done"
    assert "add_two" in agent.scratchpad
    # Meta-tools and the exposed toolbox tool are all removed once the run ends.
    tool_names = {t["name"] if isinstance(t, dict) else t.name for t in agent.tools}
    assert "expose_toolbox" not in tool_names
    assert "expose_tool" not in tool_names
    assert "add_two" not in tool_names


def test_expose_tool_single_lookup_across_toolboxes():
    responses = [
        json.dumps({"thought": "t", "actions": [{"action": "expose_tool", "action_input": {"tool_name": "beta"}}], "final_answer": None}),
        json.dumps({"thought": "t", "actions": [{"action": "beta", "action_input": {}}], "final_answer": None}),
        _final("done"),
    ]
    agent = make_test_agent(
        responses=responses,
        toolbox=[
            Toolbox("a", "toolbox a", [_tool("alpha")]),
            Toolbox("b", "toolbox b", [_tool("beta", result="beta-result")]),
        ],
        max_iterations=5,
    )
    result = agent.invoke("go")

    assert result == "done"
    assert "beta-result" in agent.scratchpad


def test_expose_toolbox_action_returns_error_for_unknown_toolbox():
    runtime = _ToolboxRuntime(toolboxes=[Toolbox("a", "toolbox a", [_tool("x")])])
    agent = DummyAgent()
    runtime.start(agent)

    result = runtime._expose_toolbox_action("does-not-exist", agent=agent)

    assert "not found" in result
    assert "a" in result


def test_expose_toolbox_action_is_idempotent_when_already_exposed():
    runtime = _ToolboxRuntime(toolboxes=[Toolbox("a", "toolbox a", [_tool("x")])])
    agent = DummyAgent()
    runtime.start(agent)

    first = runtime._expose_toolbox_action("a", agent=agent)
    tools_after_first = list(agent.tools)
    second = runtime._expose_toolbox_action("a", agent=agent)

    assert "Success" in first
    assert "already exposed" in second
    assert agent.tools == tools_after_first  # no duplicate re-add


def test_expose_tool_action_returns_error_for_unknown_tool():
    runtime = _ToolboxRuntime(toolboxes=[Toolbox("a", "toolbox a", [_tool("x")])])
    agent = DummyAgent()
    runtime.start(agent)

    result = runtime._expose_tool_action("does-not-exist", agent=agent)

    assert "not found" in result


def test_restore_returns_displaced_tool_after_exposure():
    """A toolbox tool sharing a name with a tool the agent already had
    displaces it while exposed; restoring at run end must bring the
    original back rather than leaving it gone."""
    original_x = _tool("x", result="original")
    toolbox_x = _tool("x", result="from-toolbox")
    runtime = _ToolboxRuntime(toolboxes=[Toolbox("a", "toolbox a", [toolbox_x])])
    agent = DummyAgent(tools=[original_x])

    runtime.start(agent)
    runtime._expose_toolbox_action("a", agent=agent)

    exposed_x = next(t for t in agent.tools if t["name"] == "x")
    assert exposed_x is toolbox_x

    runtime.restore(agent)

    restored_x = next(t for t in agent.tools if t["name"] == "x")
    assert restored_x is original_x


def test_restore_on_error_also_restores_displaced_tool():
    original_x = _tool("x", result="original")
    toolbox_x = _tool("x", result="from-toolbox")
    runtime = _ToolboxRuntime(toolboxes=[Toolbox("a", "toolbox a", [toolbox_x])])
    agent = DummyAgent(tools=[original_x])

    runtime.start(agent)
    runtime._expose_toolbox_action("a", agent=agent)
    runtime.restore(agent)  # restore() is unconditional -- same call on success or error

    restored_x = next(t for t in agent.tools if t["name"] == "x")
    assert restored_x is original_x


# ── Agent.add_toolbox() -- the post-construction equivalent of toolbox= ──

def test_agent_add_toolbox_registers_without_constructor_kwarg():
    """agent.add_toolbox(...) must work even when the agent was built with
    no toolbox= at all (self._toolbox starts as None)."""
    responses = [
        json.dumps({"thought": "t", "actions": [{"action": "expose_toolbox", "action_input": {"toolbox_name": "web"}}], "final_answer": None}),
        json.dumps({"thought": "t", "actions": [{"action": "search", "action_input": {}}], "final_answer": None}),
        _final("done"),
    ]
    agent = make_test_agent(responses=responses, max_iterations=5)
    agent.add_toolbox("web", "Web tools.", [_tool("search", result="results")])

    result = agent.invoke("go")

    assert result == "done"
    assert "results" in agent.scratchpad


def test_agent_add_toolbox_composes_with_constructor_toolbox_kwarg():
    responses = [
        json.dumps({"thought": "t", "actions": [{"action": "expose_toolbox", "action_input": {"toolbox_name": "db"}}], "final_answer": None}),
        json.dumps({"thought": "t", "actions": [{"action": "query", "action_input": {}}], "final_answer": None}),
        _final("done"),
    ]
    agent = make_test_agent(
        responses=responses,
        toolbox=[Toolbox("web", "web tools", [_tool("search")])],
        max_iterations=5,
    )
    agent.add_toolbox("db", "DB tools.", [_tool("query", result="rows")])

    result = agent.invoke("go")

    assert result == "done"
    assert "rows" in agent.scratchpad


def test_agent_add_toolbox_still_rejects_cross_toolbox_collision():
    agent = make_test_agent(responses=[_final("done")])
    agent.add_toolbox("a", "toolbox a", [_tool("shared")])

    with pytest.raises(ValueError, match="shared"):
        agent.add_toolbox("b", "toolbox b", [_tool("shared")])
