"""
Integration test against the real autourgos-responses package (not the
ScriptedFakeLLM double used elsewhere), pinned to verify the fix in
autourgos-responses 2.2.0.

Same bug class as tests/test_openaichat_integration.py: OpenAIResponse's
BaseLLM is re-exported from autourgos-openaichat and declares **kwargs on
invoke/ainvoke, but the concrete methods had a closed keyword signature, so
an on_before_iteration middleware hook injecting overrides raised TypeError
and crashed the whole agent run. This would fail with a TypeError on
autourgos-responses <2.2.0.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

autourgos_responses = pytest.importorskip("autourgos_responses")
from autourgos_responses import OpenAIResponse  # noqa: E402

from autourgos_agent import CallbackHandler, Agent  # noqa: E402


def _make_llm(**kwargs: Any) -> OpenAIResponse:
    with patch("autourgos_responses.response.load_openai_module") as mock_load:
        mock_load.return_value = (True, MagicMock(), MagicMock(), None)
        return OpenAIResponse(model="gpt-4o", api_key="sk-test-dummy", **kwargs)


def _mock_response_obj(text: str):
    resp = MagicMock()
    resp.output = [{"type": "message", "content": [{"text": text}]}]
    resp.usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    return resp


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
    def on_before_iteration(self, iteration: int, agent: Any = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return {"temperature": 0.0}


def test_react_agent_with_real_openairesponse_llm_and_on_before_iteration_overrides():
    import json

    llm = _make_llm()
    captured: Dict[str, Any] = {}

    def fake_attempt(client, params, label, deadline=None):
        captured.update(params)
        return _mock_response_obj(json.dumps({"thought": None, "actions": [], "final_answer": "42"}))

    llm._attempt_sync_create = fake_attempt

    handler = InjectTemperatureHandler()
    agent = Agent(llm=llm, middleware=[handler], max_iterations=5)
    agent.add_tools(_ECHO_TOOL)

    # Pre-2.2.0 this raised TypeError the moment the middleware's overrides
    # reached OpenAIResponse.invoke(). It must now complete normally.
    result = agent.invoke("what is the answer?")

    assert result == "42"
    assert captured["temperature"] == 0.0
