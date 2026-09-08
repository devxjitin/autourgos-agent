"""
Tests for Agent(history=<folder>) -- the inbuilt, direct-call history
recorder (autourgos_agent/history.py). Verifies the Markdown + JSON files
get written, cover redaction, the include_* gates, and the error path.
"""

from __future__ import annotations

import json
import os

from autourgos_agent import Agent
from autourgos_agent.testing import ScriptedToolCallLLM, make_test_agent


def _history_files(folder: str):
    names = sorted(os.listdir(folder))
    md = [n for n in names if n.endswith(".md")]
    js = [n for n in names if n.endswith(".json")]
    assert len(md) == 1 and len(js) == 1, names
    return os.path.join(folder, md[0]), os.path.join(folder, js[0])


def test_history_writes_markdown_and_json(tmp_path):
    responses = [
        json.dumps({
            "thought": "let's echo",
            "actions": [{"action": "echo", "action_input": {"text": "hi"}}],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "final answer text"}),
    ]
    agent = make_test_agent(responses=responses, history=str(tmp_path))

    result = agent.invoke("do the thing")
    assert result == "final answer text"

    md_path, json_path = _history_files(str(tmp_path))

    md_content = open(md_path, encoding="utf-8").read()
    assert "Iteration 1" in md_content
    assert "let's echo" in md_content
    assert "echo" in md_content
    assert "Final Answer" in md_content
    assert "final answer text" in md_content
    # include_query defaults to False -- the raw query must not appear
    assert "do the thing" not in md_content
    assert "[REDACTED: length=" in md_content

    logs = json.loads(open(json_path, encoding="utf-8").read())
    assert logs["final_response"] == "final answer text"
    assert logs["query"].startswith("[REDACTED: length=")
    assert len(logs["iterations"]) == 2
    assert logs["iterations"][0]["thought"] == "let's echo"


def test_history_redacts_secret_shaped_tool_output(tmp_path):
    tool = {
        "name": "leaky",
        "description": "leaks a secret",
        "parameters": {"type": "object", "properties": {}},
        "func": lambda: "here is a key: sk-abcdef1234567890",
    }
    responses = [
        json.dumps({
            "thought": None,
            "actions": [{"action": "leaky", "action_input": {}}],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "done"}),
    ]
    agent = make_test_agent(responses=responses, tools=[tool], history=str(tmp_path))
    agent.invoke("leak it")

    md_path, json_path = _history_files(str(tmp_path))
    md_content = open(md_path, encoding="utf-8").read()
    assert "sk-abcdef1234567890" not in md_content
    assert "[REDACTED]" in md_content

    logs = json.loads(open(json_path, encoding="utf-8").read())
    assert "sk-abcdef1234567890" not in json.dumps(logs)


def test_history_no_folder_means_no_files(tmp_path):
    # No history= passed -- Agent must not write anything under tmp_path,
    # and invoke() must behave exactly as without history at all.
    responses = [json.dumps({"thought": None, "actions": [], "final_answer": "ok"})]
    agent = make_test_agent(responses=responses)
    result = agent.invoke("hello")
    assert result == "ok"
    assert os.listdir(str(tmp_path)) == []


def test_history_records_error_path(tmp_path):
    # An LLM response that keeps failing to parse drives the agent into
    # AgentParseError -- history should still get a final file with the
    # error recorded, via the on_agent_error / fail() path.
    from autourgos_agent import AgentParseError

    agent = make_test_agent(
        responses=["not json"] * 5,
        history=str(tmp_path),
        max_consecutive_parse_errors=2,
    )

    try:
        agent.invoke("break it")
        assert False, "expected AgentParseError"
    except AgentParseError:
        pass

    md_path, json_path = _history_files(str(tmp_path))
    logs = json.loads(open(json_path, encoding="utf-8").read())
    assert logs["error"] is not None
    md_content = open(md_path, encoding="utf-8").read()
    assert "Final Answer" in md_content


def test_history_native_mode(tmp_path):
    llm = ScriptedToolCallLLM([
        ScriptedToolCallLLM.tool_call("echo", {"text": "hi"}, call_id="c1"),
        ScriptedToolCallLLM.final("native done"),
    ])
    agent = Agent(llm=llm, tool_calling_mode="native", history=str(tmp_path))
    agent.add_tools({
        "name": "echo",
        "description": "Echo the given text back.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
        "func": lambda text="": f"echo: {text}",
    })

    result = agent.invoke("go")
    assert result == "native done"

    md_path, json_path = _history_files(str(tmp_path))
    md_content = open(md_path, encoding="utf-8").read()
    assert "echo" in md_content
    assert "native done" in md_content

    logs = json.loads(open(json_path, encoding="utf-8").read())
    assert logs["final_response"] == "native done"
    # Native mode: no thought text is fired for the tool-call turn.
    assert logs["iterations"][0]["thought"] is None
