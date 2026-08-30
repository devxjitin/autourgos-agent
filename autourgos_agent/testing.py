"""
testing.py — Shared test fixture for autourgos-agent and its
sibling middleware packages (hcix, summarizer, preiteration, toolbox, ...).

``make_test_agent()`` builds a REAL, fully-functional ``Agent`` wired
to a small scripted fake LLM — no network calls, no mocking of the agent
itself. Sibling packages should use this instead of hand-rolling their own
fake agents, since a hand-rolled fake's shape can silently drift from the
real ``Agent`` and hide real bugs.

Example
-------
    from autourgos_agent.testing import make_test_agent

    agent = make_test_agent(responses=[
        '{"thought": "thinking", "actions": [], "final_answer": "42"}',
    ])
    result = agent.invoke("what is the answer?")
    assert result == "42"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .agent import Agent
from .base import BaseLLM, CallbackHandler, MemoryProtocol

__all__ = ["make_test_agent", "ScriptedFakeLLM", "ScriptedToolCallLLM"]


class ScriptedFakeLLM(BaseLLM):
    """
    A fake LLM that returns a pre-configured sequence of canned responses,
    one per call to invoke()/ainvoke(). No network calls.

    Each response should be raw text matching the {thought, actions,
    final_answer} JSON format the agent loop expects (see prompt.py).

    Once the scripted responses are exhausted, further calls return a
    default "final_answer" response so a misconfigured test doesn't hang
    forever inside the loop.
    """

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses: List[str] = list(responses)
        self.calls: List[Dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def _next_response(self) -> str:
        if self._responses:
            return self._responses.pop(0)
        return '{"thought": null, "actions": [], "final_answer": "done"}'

    def invoke(self, prompt: Any, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return self._next_response()

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return self._next_response()


class _FakeFunctionCall:
    """Minimal duck-type of autourgos_openaichat.FunctionCall (name/arguments/call_id) --
    defined locally instead of imported so this module stays dependency-free."""

    def __init__(self, name: str, arguments: Dict[str, Any], call_id: str) -> None:
        self.name = name
        self.arguments = arguments
        self.call_id = call_id


class _FakeToolCallResponse:
    """Minimal duck-type of autourgos_openaichat.ToolCallResponse (text/tool_calls/
    has_tool_calls/is_final_answer) -- see _FakeFunctionCall for why it's local."""

    def __init__(self, text: Optional[str] = None, tool_calls: Optional[List[_FakeFunctionCall]] = None) -> None:
        self.text = text
        self.tool_calls = list(tool_calls or [])
        self.raw = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_final_answer(self) -> bool:
        return self.text is not None and not self.tool_calls


class ScriptedToolCallLLM(BaseLLM):
    """
    A fake LLM for testing ``Agent(tool_calling_mode="native")`` --
    the native-mode counterpart to ``ScriptedFakeLLM``. Returns a
    pre-configured sequence of ``ToolCallResponse``-shaped objects, one per
    call to ``invoke_with_tools()``/``ainvoke_with_tools()``. No network
    calls, no dependency on autourgos-openaichat/autourgos-responses.

    Build responses with the ``tool_call()``/``final()`` helpers::

        from autourgos_agent.testing import ScriptedToolCallLLM

        llm = ScriptedToolCallLLM([
            ScriptedToolCallLLM.tool_call("add", {"a": 2, "b": 3}, call_id="c1"),
            ScriptedToolCallLLM.final("5"),
        ])
    """

    def __init__(self, responses: Sequence[_FakeToolCallResponse]) -> None:
        self._responses: List[_FakeToolCallResponse] = list(responses)
        self.calls: List[Dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @staticmethod
    def tool_call(name: str, arguments: Dict[str, Any], call_id: str = "call_1") -> _FakeToolCallResponse:
        """Build a response where the model wants to call one tool. For
        multiple tool calls in one turn, build the response directly:
        ``_FakeToolCallResponse(tool_calls=[...])``."""
        return _FakeToolCallResponse(tool_calls=[_FakeFunctionCall(name, arguments, call_id)])

    @staticmethod
    def calls_(specs: List[Dict[str, Any]]) -> _FakeToolCallResponse:
        """Build a response with multiple tool calls in one turn. Each spec
        is {"name":..., "arguments":..., "call_id"?: ...}."""
        return _FakeToolCallResponse(tool_calls=[
            _FakeFunctionCall(s["name"], s["arguments"], s.get("call_id", f"call_{i}"))
            for i, s in enumerate(specs)
        ])

    @staticmethod
    def final(text: str) -> _FakeToolCallResponse:
        """Build a response where the model gives its final answer."""
        return _FakeToolCallResponse(text=text)

    @staticmethod
    def empty() -> _FakeToolCallResponse:
        """Build a response with neither a final answer nor tool calls."""
        return _FakeToolCallResponse()

    def _next_response(self) -> _FakeToolCallResponse:
        if self._responses:
            return self._responses.pop(0)
        return _FakeToolCallResponse(text="done")

    def invoke(self, prompt: Any, **kwargs: Any) -> str:
        raise NotImplementedError("ScriptedToolCallLLM is for tool_calling_mode='native' only.")

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> str:
        raise NotImplementedError("ScriptedToolCallLLM is for tool_calling_mode='native' only.")

    def invoke_with_tools(self, prompt: Any, tools: List[Dict[str, Any]], **kwargs: Any) -> _FakeToolCallResponse:
        self.calls.append({"prompt": prompt, "tools": tools, "kwargs": kwargs})
        return self._next_response()

    async def ainvoke_with_tools(self, prompt: Any, tools: List[Dict[str, Any]], **kwargs: Any) -> _FakeToolCallResponse:
        self.calls.append({"prompt": prompt, "tools": tools, "kwargs": kwargs})
        return self._next_response()


_DEFAULT_TOOL: Dict[str, Any] = {
    "name": "echo",
    "description": "Echo the given text back.",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    "func": lambda text="": f"echo: {text}",
}


def make_test_agent(
    responses: Optional[Sequence[str]] = None,
    *,
    tools: Optional[List[Any]] = None,
    memory: Optional[MemoryProtocol] = None,
    middleware: Optional[List[CallbackHandler]] = None,
    max_iterations: int = 10,
    **agent_kwargs: Any,
) -> Agent:
    """
    Build and return a real, fully-functional Agent wired to a
    scripted fake LLM. No network calls.

    Parameters
    ----------
    responses : sequence of str, optional
        Canned LLM responses, one per loop iteration, as raw text in the
        {thought, actions, final_answer} JSON format. If omitted, the
        fake LLM immediately returns a final answer on the first call.
    tools : list, optional
        Tool dicts / Tool instances to attach. Defaults to a single
        harmless "echo" tool so agent.invoke() works out of the box.
    memory : MemoryProtocol, optional
        Memory backend to attach.
    middleware : list[CallbackHandler], optional
        Callback handlers to attach.
    max_iterations : int
        Passed straight through to Agent.
    **agent_kwargs
        Any other Agent constructor kwarg (verbose, system_prompt,
        approval_callback, max_execution_time, ...).

    Returns
    -------
    Agent
        A ready-to-use agent. Access ``agent.llm`` to inspect the fake
        LLM's recorded calls (``agent.llm.calls`` / ``agent.llm.call_count``).
    """
    fake_llm = ScriptedFakeLLM(responses or [])

    agent = Agent(
        llm=fake_llm,
        memory=memory,
        middleware=middleware,
        max_iterations=max_iterations,
        **agent_kwargs,
    )
    agent.add_tools(*(tools if tools is not None else [_DEFAULT_TOOL]))
    return agent
