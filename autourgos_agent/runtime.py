"""
runtime.py — Tool helpers for autourgos-react-agent.

build_tool_list   : formats tool dicts into a prompt-ready string
parse_json_object : extracts the first JSON object from LLM text
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def build_tool_list(tools: List[Any]) -> str:
    """
    Convert a list of tool dicts into a human-readable string for the prompt.

    Each tool dict must have at minimum:
        name        (str)
        description (str)

    Optional:
        parameters  (dict) — JSON-Schema style object describing inputs

    Example output::

        - get_weather: Get the current weather for a city.
            - city (string, required): The city name
            - unit (string): celsius or fahrenheit

        - search: Search the web for information.
            - query (string, required): The search query
    """
    lines: List[str] = []

    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name", "unknown")
            desc = tool.get("description", "No description provided.")
            params = tool.get("parameters", {})
        else:
            # Duck-typed tool object (not a plain dict / Tool instance) --
            # mirrors _tool_name()'s attribute fallback in base.py so a
            # non-dict tool renders into the prompt instead of crashing here
            # before the loop even starts.
            name = getattr(tool, "name", "unknown")
            desc = getattr(tool, "description", "No description provided.")
            params = getattr(tool, "parameters", {})
        lines.append(f"- {name}: {desc}")

        if isinstance(params, dict):
            props: Dict[str, Any] = params.get("properties", {})
            required: List[str] = params.get("required", [])

            for param_name, param_info in props.items():
                ptype = param_info.get("type", "any")
                pdesc = param_info.get("description", "")
                req_tag = ", required" if param_name in required else ""
                line = f"    - {param_name} ({ptype}{req_tag})"
                if pdesc:
                    line += f": {pdesc}"
                lines.append(line)

    return "\n".join(lines)


def parse_json_object(text: str) -> Dict[str, Any]:
    """
    Extract and parse the first JSON object found in *text*.

    Handles three common LLM output patterns in order:

    1. Fenced code block  —  ```json { ... } ```
    2. Bare JSON object   —  { ... } anywhere in the text
    3. Fallback           —  returns {} so callers never get an exception
    """
    if not text or not isinstance(text, str):
        return {}

    # 1 — fenced code block
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # 2 — bare JSON object (greedy: grab the widest {...} span)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            # Scan for the longest valid-JSON-parseable prefix ending in "}",
            # trying the rightmost "}" first so the common case (one valid
            # object plus trailing chatter) succeeds on the first attempt.
            # Prefix parseability isn't monotonic in "}" position (an earlier
            # "}" can parse while a later one doesn't, or vice versa), so a
            # true binary search over it isn't sound — this scan is correct
            # regardless, just worst-case O(n) attempts on adversarial input.
            raw = brace_match.group(0)
            close_positions = [i for i, ch in enumerate(raw) if ch == "}"]

            for end in reversed(close_positions):
                try:
                    return json.loads(raw[:end + 1])
                except json.JSONDecodeError:
                    continue

    return {}
