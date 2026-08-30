"""
Integration test against the real autourgos-openaichat package (not the
ScriptedFakeLLM double used elsewhere), pinned to verify the fix in
autourgos-openaichat 2.3.0.

Before 2.3.0, OpenAIChatModel.invoke()/ainvoke() had a closed keyword
signature (prompt, prompt_variables, files, image_detail) even though
AgentLoopMixin._run_loop calls self.llm.invoke(messages, **call_kwargs)
with whatever an on_before_iteration middleware hook returns. Any handler
injecting e.g. temperature= raised TypeError and crashed the whole agent
run. This test exercises that exact path end to end with a mocked openai
client (no network calls) and would fail with a TypeError on <2.3.0.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

autourgos_openaichat = pytest.importorskip("autourgos_openaichat")
from autourgos_openaichat import OpenAIChatModel  # noqa: E402

from autourgos_agent import CallbackHandler, Agent  # noqa: E402


def _make_completion(text: str):
    msg = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_llm() -> OpenAIChatModel:
    llm = OpenAIChatModel(model="gpt-4o", api_key="sk-test")
    llm._client = MagicMock()
    return llm


_ECHO_TOOL: Dict[str, Any] = {
    "name": "echo",
    "description": "Echo the given text back.",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    "func": lambda text="": f"echo: {text}",
}


class InjectTemperatureHandler(CallbackHandler):
    """Mirrors a real middleware use case: cool the model down mid-run."""

    def on_before_iteration(self, iteration: int, agent: Any = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return {"temperature": 0.0, "stop": ["Observation:"]}


def test_react_agent_with_real_openaichat_llm_and_on_before_iteration_overrides():
    llm = _make_llm()
    llm._client.chat.completions.create.return_value = _make_completion(
        json.dumps({"thought": None, "actions": [], "final_answer": "42"})
    )

    handler = InjectTemperatureHandler()
    agent = Agent(llm=llm, middleware=[handler], max_iterations=5)
    agent.add_tools(_ECHO_TOOL)

    # Pre-2.3.0 this raised TypeError: invoke() got an unexpected keyword
    # argument 'temperature' the moment the middleware's overrides reached
    # OpenAIChatModel.invoke(). It must now complete normally.
    result = agent.invoke("what is the answer?")

    assert result == "42"
    call_kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["stop"] == ["Observation:"]


@pytest.mark.asyncio
async def test_react_agent_ainvoke_with_real_openaichat_llm_and_overrides():
    llm = _make_llm()
    llm._async_client = MagicMock()
    llm._async_client.chat = MagicMock()
    llm._async_client.chat.completions = MagicMock()

    async def _fake_create(**kwargs: Any):
        return _make_completion(json.dumps({"thought": None, "actions": [], "final_answer": "done"}))

    llm._async_client.chat.completions.create = _fake_create

    handler = InjectTemperatureHandler()
    agent = Agent(llm=llm, middleware=[handler], max_iterations=5)
    agent.add_tools(_ECHO_TOOL)

    result = await agent.ainvoke("go")
    assert result == "done"
