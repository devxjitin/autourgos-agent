"""
Tests for runtime.parse_json_object, focused on:
  - the trailing-truncation fallback (binary-search prefix search)
  - correctness vs. the original O(n) linear-scan implementation
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from autourgos_agent.runtime import parse_json_object


def _old_linear_fallback(raw: str) -> Dict[str, Any]:
    """Reference re-implementation of the ORIGINAL linear fallback logic,
    used only to verify the new binary-search version produces identical
    results."""
    for end in range(len(raw) - 1, 0, -1):
        if raw[end] == "}":
            try:
                return json.loads(raw[: end + 1])
            except json.JSONDecodeError:
                continue
    return {}


def _old_parse_json_object(text: str) -> Dict[str, Any]:
    """Full reference re-implementation of the ORIGINAL parse_json_object,
    including the fenced-block and bare-brace stages, but using the
    linear fallback instead of the new binary-search one."""
    import re

    if not text or not isinstance(text, str):
        return {}

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            return _old_linear_fallback(brace_match.group(0))

    return {}


# -- fixtures: valid Agent-shaped JSON docs, truncated at every length ---------

_SAMPLE_DOCS = [
    {"thought": "I should search", "actions": [{"action": "search", "action_input": {"q": "cats"}}], "final_answer": None},
    {"thought": None, "actions": [], "final_answer": "The answer is 42."},
    {"thought": "multi step", "actions": [
        {"action": "a", "action_input": {"x": 1}},
        {"action": "b", "action_input": {"y": [1, 2, 3], "z": {"nested": True}}},
    ], "final_answer": None},
    {"thought": "unicode ✓ and \"quotes\"", "actions": [], "final_answer": "done — with an em dash"},
]


def test_binary_search_matches_linear_on_truncated_fixtures():
    for doc in _SAMPLE_DOCS:
        full = json.dumps(doc)
        # Truncate at every possible length and compare old vs new results
        # on text that is NOT itself perfectly valid JSON (so the fallback
        # path is actually exercised).
        for cut in range(1, len(full)):
            truncated = full[:cut]
            wrapped = f"Here is my response:\n{truncated}\nend"
            expected = _old_parse_json_object(wrapped)
            actual = parse_json_object(wrapped)
            assert actual == expected, (
                f"mismatch at cut={cut} doc={doc!r}\n"
                f"truncated={truncated!r}\nexpected={expected!r}\nactual={actual!r}"
            )


def test_malformed_json_fixtures():
    fixtures = [
        '{"thought": "hi", "actions": [], "final_answer": "done"',  # missing closing brace
        '{"thought": "hi", "actions": [{"action": "x", "action_input": {"y": 1}}], "final_answer": null} trailing garbage {',
        '{"a": 1, "b": {"c": 2}, "d": 3} extra text } more {not json',
        'noise before {"thought": null, "actions": [{"action": "t", "action_input": {}}], "final_answer": null} noise after',
        '{"broken": [1, 2, "unterminated string}',
        '{}',
        'not json at all',
        '',
    ]
    for text in fixtures:
        expected = _old_parse_json_object(text)
        actual = parse_json_object(text)
        assert actual == expected, f"mismatch for {text!r}: expected={expected!r} actual={actual!r}"


def test_valid_json_still_parses_directly():
    text = json.dumps({"thought": "ok", "actions": [], "final_answer": "yes"})
    assert parse_json_object(text) == {"thought": "ok", "actions": [], "final_answer": "yes"}


def test_fenced_json_still_works():
    text = '```json\n{"thought": "x", "actions": [], "final_answer": "y"}\n```'
    assert parse_json_object(text) == {"thought": "x", "actions": [], "final_answer": "y"}


def test_no_json_returns_empty_dict():
    assert parse_json_object("no json here") == {}
    assert parse_json_object("") == {}
    assert parse_json_object(None) == {}  # type: ignore[arg-type]


def test_valid_object_survives_trailing_braces_after_it():
    """
    Regression test: the binary-search fallback (replaced by a scan-from-
    the-end linear search) used to set `lo = mid + 1` on BOTH success and
    failure, so it never actually explored the lower half of the search
    space. With 3+ closing braces where the valid prefix sits earlier than
    the probed midpoints, it returned {} instead of the real object -- e.g.
    a real Agent final_answer discarded just because the model appended
    trailing chatter containing brace characters after it.
    """
    text = (
        '{"thought": null, "actions": [], "final_answer": "done"} '
        "\n\nNote: {some} extra {braces} here"
    )
    assert parse_json_object(text) == {"thought": None, "actions": [], "final_answer": "done"}

    text2 = '{"a": 1} garbage {"b": 2} more garbage {"c": 3}'
    assert parse_json_object(text2) == {"a": 1}
