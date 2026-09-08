"""
history.py — Inbuilt agent-run history, written directly from the loop.

``Agent(history="/path/to/folder")`` records every run to a Markdown + JSON
file pair under that folder. This is a direct, in-process recorder called
from AgentLoopMixin/Agent at the same points ``callback_manager.fire_*`` is
already called -- NOT a CallbackHandler/middleware. When ``history=`` is not
given, ``Agent`` uses ``_NULL_HISTORY``, whose methods are all no-ops, so
call sites never need to branch on whether history is enabled.

AgentHistoryWriter is a near-verbatim port of the old ``autourgos-history``
package's ``AgentHistoryLogger`` -- the redaction/truncation logic there is
security-sensitive and already covers real-world secret shapes, so it is
copied rather than reworked.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .runtime import parse_json_object

logger = logging.getLogger(__name__)


# ── AgentHistoryWriter ───────────────────────────────────────────────────────

class AgentHistoryWriter:
    """
    Low-level Markdown file writer for a single agent task session.

    Security
    --------
    Values whose keys match common secret patterns (api_key, token, password/
    pwd/pass, secret, authorization/auth, cookie, session, credential,
    private_key, access_key, client_secret) are replaced with [REDACTED]
    before writing.

    Values that look like secrets by shape (sk-..., Bearer ..., AKIA...,
    ghp_..., ya29..., xox[baprs]-... Slack tokens, JWTs) are also replaced
    regardless of key name.

    This is still a best-effort, pattern-based scheme -- a secret that
    matches neither a known key name nor a known value shape (e.g. a bare
    high-entropy string under an unrecognized key) will not be caught.

    Long strings (> 512 chars) are truncated with [TRUNCATED].
    """

    SENSITIVE_KEY_PATTERN = re.compile(
        r"(api[_-]?key|token|secret|password|pwd|pass|auth(?:orization)?|cookie|session|"
        r"credential|private[_-]?key|access[_-]?key|client[_-]?secret)",
        re.IGNORECASE,
    )
    # No leading `^` -- a secret is redacted wherever it appears in the string
    # (e.g. "my api_key is sk-...") not just when the entire value is nothing
    # but the secret. Matched via .sub() in _redact(), not .match().
    #
    # Covers, by provider/shape rather than by key name (so it also catches
    # secrets under an unrelated/unexpected key, e.g. {"pwd": "sk-..."}):
    # OpenAI (sk-...), AWS access key IDs (AKIA...), generic Bearer headers,
    # GitHub PATs (ghp_...), Google OAuth tokens (ya29....), Slack tokens
    # (xox[baprs]-...), and JWTs (three base64url segments, header.payload.sig).
    SENSITIVE_VALUE_PATTERN = re.compile(
        r"(sk-[A-Za-z0-9]{6,}"
        r"|AKIA[A-Z0-9]{12,}"
        r"|Bearer\s+\S+"
        r"|ghp_[A-Za-z0-9]{6,}"
        r"|ya29\.[A-Za-z0-9_\-]+"
        r"|xox[baprs]-[A-Za-z0-9\-]{6,}"
        r"|eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"
        r")",
        re.IGNORECASE,
    )

    def __init__(
        self,
        agent_name: str,
        enabled: bool,
        *,
        include_query: bool = False,
        include_tools: bool = True,
        include_observations: bool = True,
        include_final: bool = True,
    ) -> None:
        self.agent_name           = agent_name
        self.enabled              = enabled
        self.filepath: Optional[str] = None
        self.include_query        = include_query
        self.include_tools        = include_tools
        self.include_observations = include_observations
        self.include_final        = include_final

    # ── redaction ─────────────────────────────────────────────────────────────

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: "[REDACTED]" if self.SENSITIVE_KEY_PATTERN.search(str(k))
                   else self._redact(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._redact(item) for item in value]
        if isinstance(value, str):
            return self._redact_str(value)
        if value is None or isinstance(value, (int, float, bool)):
            return value
        # Any other object (dataclass, pydantic model, custom class, ...):
        # _render_json's json.dumps(..., default=str) would otherwise
        # stringify it AFTER _redact already returned it untouched here,
        # silently bypassing both the key-name and value-shape redaction
        # passes for anything that isn't a dict/list/str. Apply value-shape
        # redaction to its string form so pattern-matched secrets embedded
        # in it are still caught, instead of reaching the output raw.
        return self._redact_str(str(value))

    def _redact_str(self, value: str) -> str:
        value = self.SENSITIVE_VALUE_PATTERN.sub("[REDACTED]", value)
        return value if len(value) <= 512 else value[:512] + "... [TRUNCATED]"

    def _render_json(self, value: Any) -> str:
        try:
            return json.dumps(self._redact(value), indent=2, ensure_ascii=True, default=str)
        except Exception:
            return str(self._redact(value))

    # ── file operations ───────────────────────────────────────────────────────

    def start_task(self, query: str) -> None:
        if not self.enabled:
            return

        try:
            if not self.filepath:
                base_dir = os.path.join(os.getcwd(), "Agent History")
                os.makedirs(base_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename  = f"Task_{timestamp}_{uuid4().hex[:8]}.md"
                self.filepath = os.path.join(base_dir, filename)

            os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write(f"# Task Session: {self.agent_name}\n")
                f.write(f"**Started At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                if self.include_query:
                    f.write(f"## Initial Query\n{self._render_json(query)}\n\n")
                else:
                    f.write(f"## Initial Query\n[REDACTED: length={len(str(query))}]\n\n")
                f.write("---\n\n")
        except Exception as exc:
            logger.warning(
                "Failed to start history log for agent %r at %r: %s",
                self.agent_name, self.filepath, exc, exc_info=True,
            )
            self.enabled = False

    def log_iteration(
        self,
        iteration_num: int,
        thought: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        observations: Optional[List[Any]] = None,
    ) -> None:
        if not self.enabled or not self.filepath:
            return

        try:
            block = f"## Iteration {iteration_num}\n\n"

            if thought:
                block += f"### Thought\n{self._redact(thought)}\n\n"

            if tools:
                block += "### Action (Tools)\n"
                for t in tools:
                    t_name = t.get("tool") or t.get("name") or "Unknown tool"
                    params = t.get("params") or {}
                    params_str = (
                        self._render_json(params) if self.include_tools else "[REDACTED]"
                    )
                    block += f"**Tool:** `{t_name}`\n"
                    block += f"**Parameters:**\n```json\n{params_str}\n```\n\n"

            if observations:
                block += "### Observations\n"
                for obs in observations:
                    if isinstance(obs, dict) and "result" in obs:
                        tool_name = obs.get("tool", "Result")
                        block += f"**{tool_name}**:\n"
                        block += (
                            f"{self._render_json(obs['result'])}\n\n"
                            if self.include_observations
                            else "[REDACTED]\n\n"
                        )
                    else:
                        block += "**Result**:\n"
                        block += (
                            f"{self._render_json(obs)}\n\n"
                            if self.include_observations
                            else "[REDACTED]\n\n"
                        )

            block += "---\n\n"

            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(block)

        except Exception as exc:
            logger.warning(
                "Failed to append iteration %s to history log for agent %r at %r: %s",
                iteration_num, self.agent_name, self.filepath, exc, exc_info=True,
            )

    def log_final(self, answer: str) -> None:
        if not self.enabled or not self.filepath:
            return

        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                if self.include_final:
                    f.write(f"## Final Answer\n{self._render_json(answer)}\n")
                else:
                    f.write("## Final Answer\n[REDACTED]\n")
        except Exception as exc:
            logger.warning(
                "Failed to finalize history log for agent %r at %r: %s",
                self.agent_name, self.filepath, exc, exc_info=True,
            )


# ── _NullHistory ───────────────────────────────────────────────────────────

class _NullHistory:
    """No-op recorder used when ``Agent(history=...)`` is not set, so loop
    call sites never need to branch on whether history is enabled."""

    def start(self, query: str, agent_name: str) -> None: ...
    def begin_iteration(self, iteration: int) -> None: ...
    def record_thought(self, response_text: Optional[str]) -> None: ...
    def record_tool_start(self, tool_name: str, tool_input: Any) -> None: ...
    def record_tool_result(self, tool_name: str, result: Any) -> None: ...
    def record_tool_error(self, tool_name: str, error: BaseException) -> None: ...
    def finish(self, answer: str) -> None: ...
    def fail(self, error: BaseException) -> None: ...


_NULL_HISTORY = _NullHistory()


# ── _HistoryRecorder ─────────────────────────────────────────────────────────

class _HistoryRecorder:
    """
    Direct (non-middleware) recorder wired up by ``Agent(history=<folder>)``.

    Called straight from the loop (agent.py's invoke()/ainvoke(), base.py's
    AgentLoopMixin) at the same points ``callback_manager.fire_*`` already
    fires -- see history.py's module docstring.

    Per-run state (current iteration, buffered iteration data, JSON log) is
    held as plain instance attributes, not contextvars/RunScopedState: an
    Agent instance already rejects a second concurrent invoke()/ainvoke()
    (AgentAlreadyRunningError), so only one run is ever in flight per
    instance at a time.

    File location
    -------------
    Files land in ``<folder>/Task_<timestamp>_<uid>.md`` (+ matching
    ``.json``), where ``folder`` is the path passed to ``Agent(history=...)``.
    """

    def __init__(
        self,
        folder: str,
        include_query: bool = False,
        include_tools: bool = True,
        include_observations: bool = True,
        include_final: bool = True,
    ) -> None:
        self.folder                = folder
        self.include_query         = include_query
        self.include_tools         = include_tools
        self.include_observations  = include_observations
        self.include_final         = include_final

        self._cur_iter: Optional[int] = None
        self._iter_data: Dict[int, Dict[str, Any]] = {}
        self._json_path: Optional[str] = None
        self._writer: Optional[AgentHistoryWriter] = None
        self._logs: Dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="autourgos_agent_history",
        )

    # ── background task submission ────────────────────────────────────────────

    def _submit(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        try:
            self._executor.submit(fn, *args, **kwargs)
        except Exception as exc:
            logger.warning("Failed to schedule background history write: %s", exc, exc_info=True)

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self, query: str, agent_name: str) -> None:
        self._cur_iter  = None
        self._iter_data = {}
        self._logs = {
            "query": query, "agent_name": agent_name,
            "start_time": datetime.now().isoformat(), "end_time": None,
            "iterations": [], "final_response": None, "error": None,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_id = uuid.uuid4().hex[:8]
        md_path   = os.path.join(self.folder, f"Task_{timestamp}_{unique_id}.md")
        self._json_path = os.path.splitext(md_path)[0] + ".json"

        self._writer = AgentHistoryWriter(
            agent_name=agent_name,
            enabled=True,
            include_query=self.include_query,
            include_tools=self.include_tools,
            include_observations=self.include_observations,
            include_final=self.include_final,
        )
        self._writer.filepath = md_path
        self._writer.start_task(query)

    def begin_iteration(self, iteration: int) -> None:
        if self._cur_iter is not None:
            self._flush_iteration(self._cur_iter)
        self._cur_iter = iteration
        self._iter_data[iteration] = {
            "iteration": iteration, "thought": None, "tools": [], "observations": [],
        }

    def record_thought(self, response_text: Optional[str]) -> None:
        if self._cur_iter is None:
            return

        if response_text is None:
            # Native mode (tool_calling_mode="native"): the model's reasoning
            # text isn't available when it also calls tools in the same turn,
            # so this fires with response_text=None for that iteration. Leave
            # "thought" at None rather than str(None) ("Thought: None").
            return

        thought: Optional[str] = None
        recognized = False
        if isinstance(response_text, str):
            try:
                parsed = parse_json_object(response_text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                if parsed.get("thought"):
                    thought = parsed["thought"]
                    recognized = True
                elif parsed.get("final_answer"):
                    # Already recorded separately (and privacy-gated) via
                    # finish() -- don't duplicate it here as "thought".
                    recognized = True

        if not recognized:
            # Not recognizable {thought, ...} JSON -- it may itself BE (or
            # contain) the final answer. Only fall back to recording it
            # verbatim when include_final privacy isn't in effect.
            if self.include_final:
                thought = str(response_text)

        if thought is not None:
            self._iter_data.setdefault(self._cur_iter, {
                "iteration": self._cur_iter, "thought": None, "tools": [], "observations": [],
            })["thought"] = thought

    def record_tool_start(self, tool_name: str, tool_input: Any) -> None:
        if self._cur_iter is None:
            return
        self._iter_data.setdefault(self._cur_iter, {
            "iteration": self._cur_iter, "thought": None, "tools": [], "observations": [],
        })["tools"].append({"tool": tool_name, "params": tool_input})

    def record_tool_result(self, tool_name: str, result: Any) -> None:
        if self._cur_iter is None:
            return
        self._iter_data.setdefault(self._cur_iter, {
            "iteration": self._cur_iter, "thought": None, "tools": [], "observations": [],
        })["observations"].append({"tool": tool_name, "result": result})

    def record_tool_error(self, tool_name: str, error: BaseException) -> None:
        if self._cur_iter is None:
            return
        self._iter_data.setdefault(self._cur_iter, {
            "iteration": self._cur_iter, "thought": None, "tools": [], "observations": [],
        })["observations"].append({"tool": tool_name, "result": f"Error: {error}"})

    def finish(self, answer: str) -> None:
        if self._cur_iter is not None:
            self._flush_iteration(self._cur_iter)
        if self._writer:
            self._submit(self._write_final, self._writer, answer)

        self._logs["end_time"]       = datetime.now().isoformat()
        self._logs["final_response"] = answer
        self._logs["iterations"]     = [self._iter_data[k] for k in sorted(self._iter_data)]
        self._serialize_json()
        self.flush()

    def fail(self, error: BaseException) -> None:
        if self._writer is None:
            # start() never ran for this call (e.g. history= set but the
            # run failed before invoke()/ainvoke() fired agent_start) --
            # nothing to flush.
            return
        if self._cur_iter is not None:
            self._flush_iteration(self._cur_iter)
        self._submit(self._write_final, self._writer, f"Error: {error}")

        self._logs["end_time"]   = datetime.now().isoformat()
        self._logs["error"]      = str(error)
        self._logs["iterations"] = [self._iter_data[k] for k in sorted(self._iter_data)]
        self._serialize_json()
        self.flush()

    # ── internal I/O helpers ──────────────────────────────────────────────────

    def flush(self) -> None:
        """Block until all pending file-write tasks complete."""
        import concurrent.futures
        try:
            future = self._executor.submit(lambda: None)
            concurrent.futures.wait([future])
        except Exception:
            logger.warning("Failed to flush pending history writes", exc_info=True)

    def _flush_iteration(self, iteration_num: int) -> None:
        if not self._writer:
            return
        data = self._iter_data.get(iteration_num)
        if not data:
            return
        snapshot = {
            "thought":      data.get("thought"),
            "tools":        list(data.get("tools") or []),
            "observations": list(data.get("observations") or []),
        }
        self._submit(self._write_iteration, self._writer, iteration_num, snapshot)

    @staticmethod
    def _write_iteration(writer: AgentHistoryWriter, num: int, data: Dict[str, Any]) -> None:
        try:
            writer.log_iteration(
                iteration_num=num,
                thought=data.get("thought"),
                tools=data.get("tools"),
                observations=data.get("observations"),
            )
        except Exception as exc:
            logger.warning("Failed to write history iteration %s: %s", num, exc, exc_info=True)

    @staticmethod
    def _write_final(writer: AgentHistoryWriter, answer: str) -> None:
        try:
            writer.log_final(answer)
        except Exception as exc:
            logger.warning("Failed to write final history answer: %s", exc, exc_info=True)

    def _serialize_json(self) -> None:
        if not self._json_path:
            return
        try:
            logs_copy = copy.deepcopy(self._logs)
        except Exception:
            logs_copy = dict(self._logs)
        sanitized = self._sanitize_logs_for_output(logs_copy)
        self._submit(self._write_json, self._json_path, sanitized)

    def _sanitize_logs_for_output(self, logs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the same protections the Markdown file gets -- secret/PII
        redaction and the include_* gating -- to the JSON log too."""
        original_query = logs.get("query")
        writer = self._writer
        logs = writer._redact(logs) if writer is not None else dict(logs)

        if not self.include_query and original_query is not None:
            logs["query"] = f"[REDACTED: length={len(str(original_query))}]"
        if not self.include_final and logs.get("final_response") is not None:
            logs["final_response"] = "[REDACTED]"

        sanitized_iterations = []
        for it in logs.get("iterations") or []:
            it = dict(it)
            if not self.include_tools and it.get("tools"):
                it["tools"] = "[REDACTED]"
            if not self.include_observations and it.get("observations"):
                it["observations"] = "[REDACTED]"
            sanitized_iterations.append(it)
        logs["iterations"] = sanitized_iterations

        return logs

    @staticmethod
    def _write_json(path: str, logs: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to write JSON history log to %r: %s", path, exc, exc_info=True)

    def __del__(self) -> None:
        try:
            if hasattr(self, "_executor"):
                self._executor.shutdown(wait=False)
        except Exception:
            pass
