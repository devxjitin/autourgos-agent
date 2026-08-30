"""
Tests for BaseAgent.add_tools() duplicate-tool-name detection.

Regression coverage: previously a name collision was silently allowed --
both tools got listed in the LLM-facing tool prompt (with contradictory
descriptions), but tool_map = {name: tool} in the loop means only the most
recently added implementation with that name ever actually executes. The
earlier one becomes permanently dead with no warning. add_tools() now logs
a warning on collision and replaces the earlier tool, so the prompt shown
to the LLM matches what will actually run.
"""
from __future__ import annotations

import logging

from autourgos_agent import Agent
from autourgos_agent.testing import ScriptedFakeLLM


def _tool(name: str, description: str):
    return {"name": name, "description": description, "parameters": {}, "func": lambda: name}


def test_duplicate_tool_name_replaces_earlier_and_warns(caplog):
    agent = Agent(llm=ScriptedFakeLLM([]))
    with caplog.at_level(logging.WARNING, logger="autourgos_agent"):
        agent.add_tools(_tool("search", "v1"), _tool("search", "v2"))

    assert len(agent.tools) == 1  # earlier duplicate replaced, not kept alongside
    assert agent.tools[0]["description"] == "v2"
    assert any("search" in rec.message and "already registered" in rec.message for rec in caplog.records)


def test_unique_tool_names_no_warning(caplog):
    agent = Agent(llm=ScriptedFakeLLM([]))
    with caplog.at_level(logging.WARNING, logger="autourgos_agent"):
        agent.add_tools(_tool("search", "v1"), _tool("calculate", "v2"))

    assert len(agent.tools) == 2
    assert not any("already registered" in rec.message for rec in caplog.records)


def test_duplicate_name_across_separate_add_tools_calls_also_warns(caplog):
    agent = Agent(llm=ScriptedFakeLLM([]))
    agent.add_tools(_tool("search", "v1"))
    with caplog.at_level(logging.WARNING, logger="autourgos_agent"):
        agent.add_tools(_tool("search", "v2"))

    assert len(agent.tools) == 1
    assert agent.tools[0]["description"] == "v2"
    assert any("already registered" in rec.message for rec in caplog.records)
