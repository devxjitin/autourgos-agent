"""
runtime.py — Tool helpers for autourgos-agent.

build_tool_list      : formats tool dicts into a prompt-ready string
parse_json_object    : extracts the first JSON object from LLM text
inject_prompt_block  : prepend a text block to agent.system_prompt (or
                        prompt_template), order-independent across multiple
                        middleware
remove_prompt_block  : undo exactly one inject_prompt_block() call
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


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


# ── Prompt block injection ──────────────────────────────────────────────────
# Shared primitive for middleware that needs to prepend text into an agent's
# system_prompt (or prompt_template as a fallback for a non-Agent-shaped
# host) at runtime and later undo exactly that insertion -- e.g.
# autourgos-toolbox announcing a newly-exposed toolbox, autourgos-skills
# announcing a loaded skill, autourgos-hcix injecting a human override.
#
# Multiple such middleware can be attached to the same agent at once, in any
# registration order, and each may inject/remove several times over one run
# (toolbox: once per expose_toolbox() call; hcix: once per human interrupt).
# A "snapshot the whole string before, restore the whole string after" design
# (what every one of these packages originally did independently) is
# order-dependent: whichever middleware's on_agent_end fires *last* wins,
# clobbering any restore an earlier middleware already did, and the middleware
# that registered *second* would have snapshotted the *first* middleware's
# already-injected text as if it were the original base -- so restoring never
# actually gets back to the true original, and the leaked text grows every
# run.
#
# The fix: never snapshot or assume anything about "the original" value.
# Each caller tracks only the exact literal string ITS OWN calls inserted
# (the return value of inject_prompt_block()) and, on cleanup, removes only
# that exact substring -- correct regardless of what any other middleware
# does before, after, or in between, and regardless of registration order.

def inject_prompt_block(agent: Any, text: str, *, prepend: bool = True) -> Optional[str]:
    """
    Add ``text`` to ``agent.system_prompt`` (falling back to
    ``agent.prompt_template`` for a host that doesn't expose the former) --
    prepended (default) or appended, per ``prepend``.

    Returns the exact string this call inserted -- ``text`` alone if the
    prior value was empty/None, or ``text`` plus a ``"\\n\\n"`` separator
    (leading when appending, trailing when prepending) when combined with
    existing non-empty content -- so a later ``remove_prompt_block()`` call
    can undo precisely this insertion. Returns None if the agent exposes
    neither attribute (nothing was injected).
    """
    for attr in ("system_prompt", "prompt_template"):
        if not hasattr(agent, attr):
            continue
        current = getattr(agent, attr)
        if not current:
            setattr(agent, attr, text)
            return text
        if prepend:
            inserted = f"{text}\n\n"
            setattr(agent, attr, inserted + current)
        else:
            inserted = f"\n\n{text}"
            setattr(agent, attr, current + inserted)
        return inserted
    return None


def remove_prompt_block(agent: Any, inserted: Optional[str]) -> None:
    """
    Undo exactly one ``inject_prompt_block()`` call, given the exact string
    it returned.

    No-op if ``inserted`` is falsy, if the agent exposes neither
    ``system_prompt`` nor ``prompt_template``, or if that literal substring
    is no longer present (already removed, or the agent's prompt was reset
    externally) -- removes at most one occurrence, so a coincidental repeat
    of the same text elsewhere is left untouched.
    """
    if not inserted:
        return
    if hasattr(agent, "system_prompt") and isinstance(agent.system_prompt, str):
        agent.system_prompt = agent.system_prompt.replace(inserted, "", 1)
    elif hasattr(agent, "prompt_template") and isinstance(agent.prompt_template, str):
        agent.prompt_template = agent.prompt_template.replace(inserted, "", 1)
