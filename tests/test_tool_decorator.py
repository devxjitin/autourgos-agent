"""
Tests for the @tool decorator in autourgos-agent.

Covers:
  (a) bare @tool usage: schema inference from type hints + docstring
  (b) @tool(name=..., description=..., parameters=...) overrides
  (c) decorated function stays directly callable
  (d) end-to-end: agent.add_tools(decorated_func) works exactly like a
      hand-written tool dict, through the full Agent loop
  (e) backward compatibility: plain tool dicts still work unchanged,
      and a decorated tool + a plain dict tool can be mixed
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from autourgos_agent import Agent, Tool, tool

# -- test doubles -------------------------------------------------------------

class FakeLLM:
    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)

    def invoke(self, prompt: Any, **kwargs: Any) -> str:
        if self._responses:
            return self._responses.pop(0)
        return json.dumps({"thought": None, "actions": [], "final_answer": "done"})


# -- (a) bare @tool: schema inference -----------------------------------------

def test_bare_tool_infers_schema_from_type_hints_and_docstring():
    @tool
    def get_weather(city: str, unit: str = "celsius") -> str:
        """Get the current weather for a city.

        Args:
            city: City name, e.g. Tokyo
            unit: celsius or fahrenheit
        """
        return f"The weather in {city} is 22 degrees {unit}."

    assert isinstance(get_weather, Tool)
    assert isinstance(get_weather, dict)
    assert get_weather["name"] == "get_weather"
    assert get_weather["description"] == "Get the current weather for a city."

    params = get_weather["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["city"] == {"type": "string", "description": "City name, e.g. Tokyo"}
    assert params["properties"]["unit"]["type"] == "string"
    assert params["properties"]["unit"]["description"] == "celsius or fahrenheit"
    # only city has no default -> only city is required
    assert params["required"] == ["city"]

    assert get_weather["func"] is get_weather.func


def test_type_hint_mapping_for_non_string_types():
    @tool
    def compute(a: int, b: float, flag: bool, items: list, meta: dict) -> str:
        """Do a computation."""
        return "ok"

    props = compute["parameters"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "number"
    assert props["flag"]["type"] == "boolean"
    assert props["items"]["type"] == "array"
    assert props["meta"]["type"] == "object"
    assert compute["parameters"]["required"] == ["a", "b", "flag", "items", "meta"]


def test_optional_and_union_type_hints_unwrap_to_inner_type():
    """
    Regression test: Optional[int]/Union[int, None] used to fall back to
    "string" because _json_type_for only stripped the OUTER generic name
    ("Optional"/"Union" isn't in the type map) instead of unwrapping to the
    actual inner type. A tool with an Optional[int] param would advertise
    the wrong JSON-Schema type to the LLM.
    """
    from typing import Optional, Union

    @tool
    def set_age(age: Optional[int] = None, nickname: Union[str, None] = None, score: int | None = None):
        """Set a person's age."""
        return None

    props = set_age["parameters"]["properties"]
    assert props["age"]["type"] == "integer"
    assert props["nickname"]["type"] == "string"
    assert props["score"]["type"] == "integer"
    # Optional params with a default aren't required, regardless of the fix
    assert set_age["parameters"]["required"] == []


def test_missing_docstring_falls_back_to_default_description():
    @tool
    def no_doc(x: str) -> str:
        return x

    assert no_doc["description"] == "No description provided."
    assert no_doc["parameters"]["properties"]["x"] == {"type": "string"}


# -- (b) explicit overrides ----------------------------------------------------

def test_tool_with_explicit_overrides():
    @tool(name="calculator", description="Add two numbers together.")
    def add(a: float, b: float) -> float:
        return a + b

    assert add["name"] == "calculator"
    assert add["description"] == "Add two numbers together."
    # parameters still auto-inferred since not overridden
    assert set(add["parameters"]["properties"]) == {"a", "b"}


def test_tool_with_explicit_parameters_schema():
    custom_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "search text"}},
        "required": ["query"],
    }

    @tool(parameters=custom_schema)
    def search(query: str, limit: int = 5) -> str:
        """Search."""
        return f"results for {query}"

    assert search["parameters"] is custom_schema


def test_tool_policy_metadata_is_opt_in_and_preserved():
    def describe(call):
        return call

    @tool(describe=describe, capability="files", risk="write")
    def save(text: str) -> str:
        return text

    assert save["describe"] is describe
    assert save["capability"] == "files"
    assert save["risk"] == "write"

    @tool
    def legacy(text: str) -> str:
        return text

    assert set(legacy) == {"name", "description", "parameters", "func"}


# -- (c) decorated function remains directly callable --------------------------

def test_decorated_tool_still_directly_callable():
    @tool
    def double(x: int) -> int:
        """Double a number."""
        return x * 2

    assert double(21) == 42
    assert double["func"](21) == 42


# -- (d) end-to-end through the Agent loop -------------------------------------

def test_agent_add_tools_with_decorated_function_runs_end_to_end():
    @tool
    def echo(text: str) -> str:
        """Echo back the given text."""
        return f"echo: {text}"

    responses = [
        json.dumps({
            "thought": "I should echo",
            "actions": [{"action": "echo", "action_input": {"text": "hi"}}],
            "final_answer": None,
        }),
        json.dumps({"thought": "done", "actions": [], "final_answer": "final result"}),
    ]
    llm = FakeLLM(responses)
    agent = Agent(llm=llm, max_iterations=5)
    agent.add_tools(echo)

    result = agent.invoke("say hi")
    assert result == "final result"


# -- (e) backward compatibility / mixing ---------------------------------------

def test_plain_dict_tool_still_works_unchanged():
    def raw_add(a: float, b: float) -> float:
        return a + b

    raw_tool: Dict[str, Any] = {
        "name": "raw_add",
        "description": "Add two numbers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        "func": raw_add,
    }

    responses = [
        json.dumps({
            "thought": "add them",
            "actions": [{"action": "raw_add", "action_input": {"a": 1, "b": 2}}],
            "final_answer": None,
        }),
        json.dumps({"thought": None, "actions": [], "final_answer": "3"}),
    ]
    agent = Agent(llm=FakeLLM(responses), max_iterations=5)
    agent.add_tools(raw_tool)

    result = agent.invoke("add 1 and 2")
    assert result == "3"


def test_mixing_decorated_and_plain_dict_tools():
    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"hello {name}"

    def raw_shout(text: str) -> str:
        return text.upper()

    raw_tool = {
        "name": "shout",
        "description": "Shout text.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "func": raw_shout,
    }

    responses = [
        json.dumps({"thought": None, "actions": [], "final_answer": "mixed ok"}),
    ]
    agent = Agent(llm=FakeLLM(responses), max_iterations=5)
    agent.add_tools(greet, raw_tool)

    assert len(agent.tools) == 2
    assert {t["name"] for t in agent.tools} == {"greet", "shout"}

    result = agent.invoke("no-op")
    assert result == "mixed ok"
