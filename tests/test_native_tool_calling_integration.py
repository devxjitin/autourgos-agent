"""
Integration tests for Agent(tool_calling_mode="native") against the
real autourgos-openaichat and autourgos-responses packages (mocked client,
no network calls) -- verifies invoke_with_tools()/ainvoke_with_tools()
actually compose correctly end to end: request/response shapes, multi-turn
message-list construction, and the concurrent tool-execution path.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from autourgos_agent import CallbackHandler, Agent

_ADD_TOOL: Dict[str, Any] = {
    "name": "add",
    "description": "Add two numbers.",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    },
    "func": lambda a, b: a + b,
}


# -- autourgos-openaichat (Chat Completions tool-call shape) --------------------------

autourgos_openaichat = pytest.importorskip("autourgos_openaichat")
from autourgos_openaichat import OpenAIChatModel  # noqa: E402


def _chat_tool_call_completion(name: str, args: Dict[str, Any], call_id: str = "call_1"):
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    tc = SimpleNamespace(id=call_id, function=fn)
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    return SimpleNamespace(choices=[choice], usage=usage)


def _chat_text_completion(text: str):
    msg = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_native_mode_with_real_openaichat_llm():
    llm = OpenAIChatModel(model="gpt-4o", api_key="sk-test")
    llm._client = MagicMock()
    llm._client.chat.completions.create.side_effect = [
        _chat_tool_call_completion("add", {"a": 3, "b": 4}),
        _chat_text_completion("7"),
    ]

    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5)
    agent.add_tools(_ADD_TOOL)
    result = agent.invoke("what is 3+4?")

    assert result == "7"
    # second call's message history must carry the assistant tool_calls
    # message and the matching tool-role result, in the standard shape
    second_call_messages = llm._client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert second_call_messages[-2]["role"] == "assistant"
    assert second_call_messages[-2]["tool_calls"][0]["function"]["name"] == "add"
    assert second_call_messages[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "7"}


class _InjectTemperatureHandler(CallbackHandler):
    """Mirrors test_openaichat_integration.py's InjectTemperatureHandler."""

    def on_before_iteration(self, iteration, agent=None, **kwargs):
        return {"temperature": 0.0}


def test_native_mode_on_before_iteration_overrides_reach_the_api_call():
    """
    Regression test: invoke_with_tools() used to silently drop any keyword
    besides tool_choice/files/image_detail -- an on_before_iteration
    middleware hook's returned overrides reached invoke_with_tools() but
    never affected the actual request. Fixed in autourgos-openaichat 2.3.1.
    """
    llm = OpenAIChatModel(model="gpt-4o", api_key="sk-test")
    llm._client = MagicMock()
    llm._client.chat.completions.create.side_effect = [
        _chat_tool_call_completion("add", {"a": 1, "b": 1}),
        _chat_text_completion("2"),
    ]

    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5, middleware=[_InjectTemperatureHandler()])
    agent.add_tools(_ADD_TOOL)
    agent.invoke("1+1?")

    first_call_kwargs = llm._client.chat.completions.create.call_args_list[0].kwargs
    assert first_call_kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_native_mode_async_with_real_openaichat_llm():
    llm = OpenAIChatModel(model="gpt-4o", api_key="sk-test")
    llm._async_client = MagicMock()
    llm._async_client.chat = MagicMock()
    llm._async_client.chat.completions = MagicMock()

    responses = [
        _chat_tool_call_completion("add", {"a": 1, "b": 2}),
        _chat_text_completion("3"),
    ]

    async def _fake_create(**kwargs: Any):
        return responses.pop(0)

    llm._async_client.chat.completions.create = _fake_create

    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5)
    agent.add_tools(_ADD_TOOL)
    result = await agent.ainvoke("1+2?")
    assert result == "3"


# -- autourgos-responses (Responses API tool-call shape) -----------------------------

autourgos_responses = pytest.importorskip("autourgos_responses")
from autourgos_responses import OpenAIResponse  # noqa: E402


def _responses_tool_call(name: str, args: Dict[str, Any], call_id: str = "call_1"):
    r = MagicMock()
    r.output = [{"type": "function_call", "name": name, "arguments": json.dumps(args), "call_id": call_id}]
    r.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return r


def _responses_text(text: str):
    r = MagicMock()
    r.output = [{"type": "message", "content": [{"text": text}]}]
    r.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return r


def _make_response_llm(**kwargs: Any) -> OpenAIResponse:
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model="gpt-4o", api_key="sk-test-dummy", **kwargs)


def test_native_mode_with_real_openairesponse_llm():
    llm = _make_response_llm()
    responses = [_responses_tool_call("add", {"a": 3, "b": 4}), _responses_text("7")]
    llm._attempt_sync_create = MagicMock(side_effect=lambda client, params, label, deadline=None: responses.pop(0))

    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5)
    agent.add_tools(_ADD_TOOL)
    result = agent.invoke("what is 3+4?")
    assert result == "7"


def test_native_mode_concurrent_tool_calls_with_real_openairesponse_llm():
    import time

    def slow_add(a, b):
        time.sleep(0.1)
        return a + b

    tools = [
        {"name": "add", "description": "add", "parameters": {}, "func": slow_add},
        {"name": "add2", "description": "add", "parameters": {}, "func": slow_add},
        {"name": "add3", "description": "add", "parameters": {}, "func": slow_add},
    ]

    llm = _make_response_llm()
    multi_call = MagicMock()
    multi_call.output = [
        {"type": "function_call", "name": "add", "arguments": '{"a": 1, "b": 1}', "call_id": "c0"},
        {"type": "function_call", "name": "add2", "arguments": '{"a": 2, "b": 2}', "call_id": "c1"},
        {"type": "function_call", "name": "add3", "arguments": '{"a": 3, "b": 3}', "call_id": "c2"},
    ]
    multi_call.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    responses = [multi_call, _responses_text("done")]
    llm._attempt_sync_create = MagicMock(side_effect=lambda client, params, label, deadline=None: responses.pop(0))

    agent = Agent(llm=llm, tool_calling_mode="native", max_iterations=5)
    agent.add_tools(*tools)

    start = time.monotonic()
    result = agent.invoke("run 3 tools")
    elapsed = time.monotonic() - start

    assert result == "done"
    assert elapsed < 0.25  # ~0.1s if concurrent, ~0.3s if sequential
