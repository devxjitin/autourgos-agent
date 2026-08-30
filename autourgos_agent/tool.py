'''
tool.py -- @tool decorator for autourgos-agent.

Lets you define a tool as a plain, type-hinted function instead of hand
-writing the JSON-Schema dict that Agent.add_tools() expects::

    from autourgos_agent import tool

    @tool
    def get_weather(city: str, unit: str = "celsius") -> str:
        """Get the current weather for a city.

        Args:
            city: City name, e.g. Tokyo
            unit: celsius or fahrenheit
        """
        return f"The weather in {city} is 22 degrees {unit}."

    agent.add_tools(get_weather)

The decorator returns a `Tool` -- a dict subclass carrying the exact same
`name` / `description` / `parameters` / `func` shape the agent loop has
always expected from a plain tool dict, so it is a strict superset of the
existing convention: hand-written tool dicts keep working unchanged, and
`Tool` instances remain directly callable like the original function
(`get_weather("Tokyo")` still works outside the agent).

Parameter types are inferred from type hints (str/int/float/bool/list/dict
map to the matching JSON-Schema type; anything else defaults to "string").
Parameter descriptions are best-effort parsed from a Google-style
``Args:``/``Arguments:``/``Parameters:`` docstring section when present.
Any inferred field can be overridden explicitly via the decorator's
keyword arguments.
'''

from __future__ import annotations

import functools
import inspect
import re
import types as _types
import typing
from typing import Any, Callable, Dict, List, Optional

__all__ = ["tool", "Tool"]

_JSON_TYPE_MAP: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# Mirror of _JSON_TYPE_MAP keyed by type name, used when annotations arrive
# as strings -- e.g. any module using `from __future__ import annotations`
# (PEP 563) stringifies all annotations, so `inspect.signature(...).annotation`
# is the literal text "int" rather than the `int` type object.
_JSON_TYPE_NAME_MAP: Dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "List": "array",
    "Dict": "object",
}

_SECTION_HEADER_RE = re.compile(r"^(Args|Arguments|Parameters)\s*:?\s*$", re.IGNORECASE)
_NEXT_SECTION_RE = re.compile(
    r"^(Returns?|Raises|Yields?|Examples?|Note|Notes)\s*:?\s*$", re.IGNORECASE
)
_PARAM_LINE_RE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def _split_top_level(text: str) -> List[str]:
    """Split on commas, ignoring ones nested inside [...] (e.g. Dict[str, int])."""
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _unwrap_optional(annotation: Any) -> Any:
    """
    Unwrap Optional[X] / Union[X, None] / "X | None" down to X, so the real
    inner type drives the JSON-Schema type instead of the wrapper — e.g.
    Optional[int] should infer "integer", not fall back to "string" just
    because "Optional"/"Union" aren't themselves in the type maps.

    Handles both stringified annotations (any module using
    `from __future__ import annotations` turns these into plain text) and
    live typing objects, one level deep — a further-nested Optional (e.g.
    inside a List[...]) is left alone, since only the top-level type drives
    the JSON-Schema "type" field here.
    """
    if isinstance(annotation, str):
        text = annotation.strip()
        if "|" in text and "[" not in text:
            parts = [p.strip() for p in text.split("|")]
            non_none = [p for p in parts if p not in ("None", "NoneType")]
            if len(non_none) == 1:
                return _unwrap_optional(non_none[0])
            return text
        base_name = text.split("[", 1)[0].strip()
        if base_name in ("Optional", "Union") and text.endswith("]"):
            inner = text[text.index("[") + 1 : -1]
            non_none = [a.strip() for a in _split_top_level(inner) if a.strip() not in ("None", "NoneType")]
            if len(non_none) == 1:
                return _unwrap_optional(non_none[0])
        return text

    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is getattr(_types, "UnionType", None):
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _unwrap_optional(non_none[0])
    return annotation


def _json_type_for(annotation: Any) -> str:
    annotation = _unwrap_optional(annotation)
    if isinstance(annotation, str):
        # Strip generic parameters, e.g. "List[str]" -> "List"
        base_name = annotation.split("[", 1)[0].strip()
        return _JSON_TYPE_NAME_MAP.get(base_name, "string")
    return _JSON_TYPE_MAP.get(annotation, "string")


def _parse_docstring_param_descriptions(doc: Optional[str]) -> Dict[str, str]:
    """Best-effort extraction of name/description lines from an Args section."""
    descriptions: Dict[str, str] = {}
    if not doc:
        return descriptions

    in_args_section = False
    for raw_line in doc.splitlines():
        line = raw_line.strip()
        if _SECTION_HEADER_RE.match(line):
            in_args_section = True
            continue
        if not in_args_section:
            continue
        if not line:
            continue
        if _NEXT_SECTION_RE.match(line):
            break
        match = _PARAM_LINE_RE.match(line)
        if match:
            descriptions[match.group(1)] = match.group(2).strip()
    return descriptions


def _infer_parameters(func: Callable[..., Any]) -> Dict[str, Any]:
    signature = inspect.signature(func)
    param_docs = _parse_docstring_param_descriptions(inspect.getdoc(func))

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in signature.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
        prop: Dict[str, Any] = {"type": _json_type_for(annotation)}
        if param_name in param_docs:
            prop["description"] = param_docs[param_name]
        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {"type": "object", "properties": properties, "required": required}


def _infer_description(func: Callable[..., Any]) -> str:
    doc = inspect.getdoc(func)
    if not doc:
        return "No description provided."
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "No description provided."


class Tool(dict):
    """
    Dict-shaped tool spec produced by the @tool decorator.

    Behaves exactly like the plain {"name": ..., "description": ...,
    "parameters": ..., "func": ...} dicts Agent has always accepted
    (same keys, same tool["name"] / tool.get("func") access patterns used
    by _execute_tool/build_tool_list), while also remaining directly
    callable so the wrapped function can still be used on its own.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.func = func
        resolved_name = name or getattr(func, "__name__", "tool")
        resolved_description = description if description is not None else _infer_description(func)
        resolved_parameters = parameters if parameters is not None else _infer_parameters(func)

        super().__init__(
            name=resolved_name,
            description=resolved_description,
            parameters=resolved_parameters,
            func=func,
        )

        try:
            functools.update_wrapper(self, func, updated=())
        except Exception:
            pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Tool name={self.get('name')!r}>"


def tool(
    func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Any:
    '''
    Turn a plain, type-hinted function into an Autourgos tool spec.

    Usable bare::

        @tool
        def add(a: float, b: float) -> float:
            """Add two numbers together."""
            return a + b

    or with overrides::

        @tool(name="calculator", description="Add two numbers.")
        def add(a: float, b: float) -> float:
            return a + b

    The result is a Tool (a dict subclass) ready to pass straight into
    agent.add_tools(...) -- no manual JSON-Schema required.
    '''

    def decorator(f: Callable[..., Any]) -> Tool:
        return Tool(f, name=name, description=description, parameters=parameters)

    if func is not None and callable(func):
        return decorator(func)
    return decorator
