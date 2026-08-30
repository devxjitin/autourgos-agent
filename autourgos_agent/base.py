"""
base.py — Self-contained base classes for autourgos-agent.

Inlines BaseLLM, BaseAgent, AgentLoopMixin, CallbackManager, and all
protocols so the package has zero dependency on autourgos-core.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeoutError
from typing import Any, Callable, Dict, List, Optional, Tuple

from .runtime import build_tool_list

_logger = logging.getLogger("autourgos_agent")


# ── Exceptions ────────────────────────────────────────────────────────────────
# Loop stop-conditions used to be signaled by returning a "[Tag] message"
# string from invoke()/ainvoke() instead of raising. That meant callers had
# to string-sniff the result (`result.startswith("[Max Iterations]")`) to
# tell a real answer apart from the agent giving up, which is fragile (a
# model could in principle produce an answer starting with the same tag) and
# undiscoverable (no type to catch or document). These are now raised
# instead, and propagate out of invoke()/ainvoke() like any other exception
# (agent.py's invoke()/ainvoke() already wrap the loop call in a
# try/except Exception that fires on_agent_error and re-raises, so no
# special-casing is needed there for these to be reported the same way a
# tool/LLM crash already was).

class AgentError(RuntimeError):
    """Base class for all agent-loop stop conditions raised by autourgos-agent."""


class AgentTimeoutError(AgentError):
    """Raised when max_execution_time is exceeded between iterations."""

    def __init__(self, max_execution_time: float) -> None:
        self.max_execution_time = max_execution_time
        super().__init__(f"Agent stopped after {max_execution_time}s.")


class AgentMaxIterationsError(AgentError):
    """Raised when the loop reaches max_iterations without a final answer."""

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        super().__init__(f"Agent stopped after {max_iterations} iterations without a final answer.")


class AgentParseError(AgentError):
    """Raised when the LLM's response couldn't be parsed as valid JSON,
    max_consecutive_parse_errors times in a row (prompt mode only)."""

    def __init__(self, attempts: int, last_response: str) -> None:
        self.attempts = attempts
        self.last_response = last_response
        super().__init__(
            f"Could not parse a valid JSON response after {attempts} attempts. "
            f"Last response:\n{last_response}"
        )


class AgentLLMError(AgentError):
    """Raised when the underlying LLM call itself raises."""

    def __init__(self, original: BaseException) -> None:
        self.original = original
        super().__init__(str(original))


class AgentEmptyResponseError(AgentError):
    """Raised when the LLM returns neither a final answer nor tool calls,
    max_consecutive_parse_errors times in a row (native mode only)."""

    def __init__(self, consecutive_empty: int) -> None:
        self.consecutive_empty = consecutive_empty
        super().__init__(
            f"LLM returned neither a final answer nor tool calls, "
            f"{consecutive_empty} time(s) in a row."
        )


# ── Protocols ─────────────────────────────────────────────────────────────────

class CallbackHandler:
    """
    Base class for agent middleware / event hooks.

    Sub-class and override the methods you care about.  Unused methods
    are no-ops so you never have to implement every hook.

    11 hooks total:
      - on_agent_start, on_agent_end, on_agent_error
      - on_tool_start, on_tool_end, on_tool_error
      - on_iteration_start, on_before_iteration, on_iteration, on_llm_end
      - on_parse_error

    All hooks may receive an ``agent=<Agent instance>`` kwarg.
    Older handlers written against the original 6-hook interface
    (without an ``agent`` parameter) continue to work unmodified —
    CallbackManager falls back to calling handlers without ``agent=``
    if their signature doesn't accept it.
    """

    def on_agent_start(self, query: str, agent: Any = None, **kwargs: Any) -> None:
        pass

    def on_agent_end(self, result: str, agent: Any = None, **kwargs: Any) -> None:
        pass

    def on_agent_error(self, error: Exception, agent: Any = None, **kwargs: Any) -> None:
        pass

    def on_tool_start(self, tool_name: str, tool_input: Dict[str, Any], agent: Any = None, **kwargs: Any) -> None:
        pass

    def on_tool_end(self, tool_name: str, result: str, agent: Any = None, **kwargs: Any) -> None:
        pass

    def on_tool_error(self, tool_name: str, error: Exception, agent: Any = None, **kwargs: Any) -> None:
        pass

    def on_iteration_start(self, iteration: int, agent: Any = None, **kwargs: Any) -> None:
        pass

    def on_before_iteration(self, iteration: int, agent: Any = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """
        Called once per loop iteration, just before the LLM is invoked.

        If this returns a dict, its keys/values are merged into the
        extra kwargs passed to ``self.llm.invoke()``/``ainvoke()`` for
        THAT iteration only (not persisted to later iterations). When
        multiple handlers return dicts, later handlers override earlier
        ones on key conflicts. Return None (the default) for no-op.
        """
        return None

    def on_iteration(self, iteration: int, thought: Optional[str], agent: Any = None, **kwargs: Any) -> None:
        pass

    def on_llm_end(self, response: Any, agent: Any = None, **kwargs: Any) -> None:
        """Called after each LLM call with the extracted response text.

        ``**kwargs`` carries whatever the LLM wrapper's raw response
        exposes: ``raw`` (the untouched raw response/dict) plus, when
        available, ``provider_used``, ``input_tokens``, ``output_tokens``,
        ``total_cost``, ``latency_ms`` (autourgos-openaichat/-responses
        dict shape) or ``total_tokens`` (from a native SDK response's
        ``.usage``). A cost/usage-tracking middleware should read these
        instead of reaching into ``agent.llm``'s internal attributes.
        """
        pass

    def on_parse_error(self, iteration: int, raw_response: str, agent: Any = None, **kwargs: Any) -> None:
        pass


class MemoryProtocol(ABC):
    """Abstract interface for memory backends.

    ``add_agent_message`` is the canonical method name, matching the
    ``autourgos-memory`` family's ``BaseMemory`` interface (buffer-memory,
    local-memory, semantic-memory, summary-memory, token-memory all
    implement it). ``add_assistant_message`` is accepted as a legacy alias
    for any custom memory object built against react-agent's older duck
    type — see ``_record_agent_message`` below, which tries both names.
    """

    @abstractmethod
    def add_user_message(self, message: str) -> None:
        ...

    @abstractmethod
    def add_agent_message(self, message: str) -> None:
        ...

    @abstractmethod
    def get_history(self) -> List[Dict[str, str]]:
        ...


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it's awaitable, otherwise return it as-is.

    Lets async loops accept either a plain sync approval_callback (existing
    behavior, called synchronously — still blocks the event loop for the
    duration of the call, same as before) or an async one (e.g. `async def
    approve(name, input): await slack_prompt(...)`) without needing two
    separate approval_callback parameters.
    """
    if inspect.isawaitable(value):
        return await value
    return value


def _call_sync_approval(
    approval_callback: Callable[[str, Dict[str, Any]], Any],
    tool_name: str,
    tool_input: Dict[str, Any],
) -> Any:
    """Call ``approval_callback`` from a sync loop (_run_loop / _run_loop_native).

    An async ``def approve(...): ...`` callback returns a coroutine object
    without ever running its body -- and a coroutine object is always
    truthy, so the tool would silently be approved regardless of what the
    callback actually decides, plus the coroutine leaks unawaited. Fail
    fast with a clear message instead, pointing at the fix (use ainvoke()
    with an async approval_callback, per _maybe_await's async support).
    """
    result = approval_callback(tool_name, tool_input)
    if inspect.isawaitable(result):
        raise TypeError(
            f"approval_callback returned an awaitable ({result!r}) but the agent is "
            f"running synchronously (invoke()). An async approval_callback is only "
            f"supported with ainvoke(). Use a plain sync callback with invoke(), or "
            f"call agent.ainvoke() instead."
        )
    return result


def _tool_name(tool: Any) -> Optional[str]:
    """Duck-typed tool name lookup: plain dicts and Tool (dict subclass) alike."""
    return tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)


def _tool_func(tool: Any) -> Optional[Callable[..., Any]]:
    """Duck-typed tool callable lookup, mirroring _tool_name: plain dicts /
    Tool instances via "func"/"function" keys, or a non-dict tool object's
    .func/.function attribute -- so a tool built without dict access (only
    reachable via _tool_name's tool_map lookup) can actually be executed
    instead of failing on tool.get() not existing."""
    if isinstance(tool, dict):
        return tool.get("func") or tool.get("function")
    return getattr(tool, "func", None) or getattr(tool, "function", None)


def _default_should_retry(exc: BaseException) -> bool:
    """Default llm_retry_on predicate: retry everything except
    NotImplementedError, which signals a config error (the LLM doesn't
    support invoke_with_tools()/ainvoke_with_tools() at all) rather than a
    transient failure -- retrying it would just delay the clearer
    RuntimeError _wrap_unsupported_native_error raises for it.
    """
    return not isinstance(exc, NotImplementedError)


def _default_token_counter(text: str) -> int:
    """Rough token-count approximation (~4 chars/token, the common rule of
    thumb for English text) used when max_scratchpad_tokens is set but no
    real tokenizer was supplied via token_counter=. This is a soft-budget
    guard, not exact -- pass e.g. `token_counter=lambda t:
    len(tiktoken.encoding_for_model(model).encode(t))` for precision,
    especially for non-English text or code, where chars-per-token can
    differ a lot from the English-prose rule of thumb this falls back to.
    """
    return max(1, len(text) // 4)


def _trim_to_token_budget(text: str, max_tokens: int, counter: Callable[[str], int]) -> str:
    """Binary-search the longest tail of `text` whose token count (per
    `counter`) fits in `max_tokens` once the "[...earlier steps
    trimmed...]" prefix is accounted for, and return prefix + that tail.

    Binary search (not a linear scan) because `counter` may be a real
    tokenizer call, not just len() -- O(log n) calls keeps this cheap even
    for a large scratchpad. Token count isn't guaranteed strictly
    monotonic in string length for every possible tokenizer (a cut can
    occasionally merge/split a token differently), so this can be off by a
    token or two at the boundary -- acceptable for a soft trim guard, and
    the search still converges since it's monotonic enough in practice for
    real tokenizers and for the default char-based approximation.
    """
    prefix = "[...earlier steps trimmed...]\n"
    prefix_tokens = counter(prefix)
    if prefix_tokens >= max_tokens:
        return prefix

    budget = max_tokens - prefix_tokens
    lo, hi, best = 0, len(text), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if counter(text[-mid:]) <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return prefix + text[-best:] if best else prefix


def _record_agent_message(memory: Any, message: str) -> None:
    """Store the agent's final answer in ``memory``, tolerating either the
    canonical ``add_agent_message`` (autourgos-memory family) or the legacy
    ``add_assistant_message`` name some custom duck-typed memory objects use.
    """
    if hasattr(memory, "add_agent_message"):
        memory.add_agent_message(message)
    elif hasattr(memory, "add_assistant_message"):
        memory.add_assistant_message(message)
    else:
        raise AttributeError(
            "memory object has neither add_agent_message() nor "
            "add_assistant_message() — cannot record the agent's reply"
        )


def _get_memory_context(memory: Any, query: str) -> str:
    """Best-effort retrieval of prior conversation history from ``memory``,
    to actually feed it back into the prompt/messages the LLM sees.

    Tolerates two different, incompatible conventions that exist across the
    Autourgos ecosystem: the autourgos-memory family's ``BaseMemory``
    (``format_for_llm()``/legacy ``get_context()``) and this package's own
    ``MemoryProtocol`` (``get_history()`` -> list of ``{role, content}``
    dicts, per the README's hand-rolled memory example). Without this, a
    memory object only ever gets written to (add_user_message/
    add_agent_message) and never read back, so the agent could never
    actually recall anything from it.
    """
    if memory is None:
        return ""

    fmt = getattr(memory, "format_for_llm", None)
    if callable(fmt):
        try:
            return fmt(query) or ""
        except TypeError:
            return fmt() or ""

    ctx = getattr(memory, "get_context", None)
    if callable(ctx):
        try:
            return ctx(query) or ""
        except TypeError:
            return ctx() or ""

    history_fn = getattr(memory, "get_history", None)
    if callable(history_fn):
        history = history_fn() or []
        if not history:
            return ""
        rendered: List[str] = []
        for m in history:
            if isinstance(m, dict):
                rendered.append(f"{m.get('role', '')}: {m.get('content', '')}")
            elif isinstance(m, (tuple, list)) and len(m) == 2:
                rendered.append(f"{m[0]}: {m[1]}")
            else:
                rendered.append(str(m))
        lines = "\n".join(rendered)
        return f"\n--- Previous Conversation Context ---\n{lines}\n--------------------------------------\n"

    return ""


# ── CallbackManager ────────────────────────────────────────────────────────────

class CallbackManager:
    """Fires lifecycle events to all registered handlers."""

    def __init__(self, handlers: Optional[List[CallbackHandler]] = None) -> None:
        self._handlers: List[CallbackHandler] = list(handlers or [])

    def add_handler(self, handler: CallbackHandler) -> None:
        self._handlers.append(handler)

    def _fire(self, method: str, *args: Any, **kwargs: Any) -> None:
        for h in self._handlers:
            fn = getattr(h, method, None)
            if not callable(fn):
                continue
            try:
                try:
                    fn(*args, **kwargs)
                except TypeError:
                    # Backward compatibility: older handlers may define a
                    # narrower signature that doesn't accept kwargs such
                    # as `agent=`. Retry without them.
                    if "agent" in kwargs:
                        retry_kwargs = {k: v for k, v in kwargs.items() if k != "agent"}
                        fn(*args, **retry_kwargs)
                    else:
                        raise
            except Exception:
                _logger.debug(
                    "Callback handler %s raised in %s",
                    type(h).__name__,
                    method,
                    exc_info=True,
                )

    def fire_agent_start(self, query: str, agent: Any = None, **kw: Any) -> None:
        self._fire("on_agent_start", query, agent=agent, **kw)

    def fire_agent_end(self, result: str, agent: Any = None, **kw: Any) -> None:
        self._fire("on_agent_end", result, agent=agent, **kw)

    def fire_agent_error(self, error: Exception, agent: Any = None, **kw: Any) -> None:
        self._fire("on_agent_error", error, agent=agent, **kw)

    def fire_tool_start(self, tool_name: str, tool_input: Dict[str, Any], agent: Any = None, **kw: Any) -> None:
        self._fire("on_tool_start", tool_name, tool_input, agent=agent, **kw)

    def fire_tool_end(self, tool_name: str, result: str, agent: Any = None, **kw: Any) -> None:
        self._fire("on_tool_end", tool_name, result, agent=agent, **kw)

    def fire_tool_error(self, tool_name: str, error: Exception, agent: Any = None, **kw: Any) -> None:
        self._fire("on_tool_error", tool_name, error, agent=agent, **kw)

    def fire_iteration_start(self, iteration: int, agent: Any = None, **kw: Any) -> None:
        self._fire("on_iteration_start", iteration, agent=agent, **kw)

    def fire_before_iteration(self, iteration: int, agent: Any = None, **kw: Any) -> Dict[str, Any]:
        """
        Calls on_before_iteration on every handler and merges any dicts
        they return. Later handlers override earlier ones on key conflict.
        Handlers that return None (or don't implement the hook) contribute
        nothing.
        """
        merged: Dict[str, Any] = {}
        for h in self._handlers:
            fn = getattr(h, "on_before_iteration", None)
            if not callable(fn):
                continue
            try:
                try:
                    result = fn(iteration, agent=agent, **kw)
                except TypeError:
                    result = fn(iteration, **kw)
            except Exception:
                _logger.debug(
                    "Callback handler %s raised in on_before_iteration",
                    type(h).__name__,
                    exc_info=True,
                )
                continue
            if isinstance(result, dict):
                merged.update(result)
        return merged

    def fire_iteration(self, iteration: int, thought: Optional[str], agent: Any = None, **kw: Any) -> None:
        self._fire("on_iteration", iteration, thought, agent=agent, **kw)

    def fire_llm_end(self, response: Any, agent: Any = None, **kw: Any) -> None:
        self._fire("on_llm_end", response, agent=agent, **kw)

    def fire_parse_error(self, iteration: int, raw_response: str, agent: Any = None, **kw: Any) -> None:
        self._fire("on_parse_error", iteration, raw_response, agent=agent, **kw)


# ── BaseLLM ────────────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    """
    Minimal abstract interface that any LLM wrapper must satisfy.

    Both autourgos-openaichat (OpenAIChatModel) and
    autourgos-responses (OpenAIResponse) already implement this
    interface, so you can pass either one to Agent.

    Any other object with .invoke() / .ainvoke() also works — the
    agent uses duck typing at runtime.
    """

    @abstractmethod
    def invoke(self, prompt: Any, **kwargs: Any) -> Any:
        """Synchronous generation. Returns str or metadata dict."""
        ...

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> Any:
        """
        Async generation.  Default implementation runs invoke() in a
        thread-pool so sync-only wrappers still work with ainvoke.
        Override this in your LLM class for a true async path.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.invoke(prompt, **kwargs))


# ── AgentLoopMixin ─────────────────────────────────────────────────────────────

class AgentLoopMixin:
    """
    Provides _run_loop (sync) and _arun_loop (async) for Agent agents.

    Expects the host class to have:
      - self.llm               — BaseLLM instance
      - self.tools             — list of tool dicts
      - self.prompt_template   — str with {tool_list}, {previous_context}, {user_input}
      - self.system_prompt     — str (may be empty)
      - self.memory            — MemoryProtocol | None
      - self.callback_manager  — CallbackManager
      - self.logger            — AgentLogger
      - self.max_execution_time — float | None
      - self.MAX_CONSECUTIVE_PARSE_ERRORS — int
      - self.MAX_SCRATCHPAD_CHARS         — int
      - self.MAX_TOOL_OUTPUT_CHARS        — int
      - self._parser(response_text)       — returns (thought, actions, final_answer)

    Also sets, as live externally-readable instance attributes on the host:
      - self.scratchpad     — str, updated in place every iteration
      - self.current_query  — str, set once per invoke()/ainvoke() call
    See the "Middleware integration contract" section of README.md.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_messages(self, prompt_text: str) -> Any:
        """Wrap the rendered prompt in messages list if a system prompt exists."""
        system_prompt: str = getattr(self, "system_prompt", "")
        if system_prompt:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt_text},
            ]
        return prompt_text

    def _extract_text(self, raw: Any) -> str:
        """Normalise LLM output to a plain string."""
        if isinstance(raw, dict):
            return raw.get("response", str(raw))
        return str(raw)

    def _extract_llm_metadata(self, raw: Any) -> Dict[str, Any]:
        """Best-effort usage/cost/latency extraction from a raw LLM response.

        ``on_llm_end`` used to only ever see the extracted text, so cost
        trackers and budget middleware had no callback-contract way to see
        tokens/cost/latency and had to reach into LLM-wrapper-internal
        attributes instead. This pulls out whatever the wrapper's raw
        response exposes (autourgos-openaichat/autourgos-responses' dict
        shape, or a `.usage` object from a native SDK response) so it can
        be passed through as callback kwargs. Returns {} if raw exposes
        neither shape.
        """
        if isinstance(raw, dict):
            keys = ("provider_used", "input_tokens", "output_tokens", "total_cost", "latency_ms")
            return {k: raw[k] for k in keys if k in raw}
        usage = getattr(raw, "usage", None)
        if usage is not None:
            return {
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return {}

    def _validate_tool_args(self, func: Callable[..., Any], tool_input: Dict[str, Any]) -> Optional[str]:
        """Bind ``tool_input`` against ``func``'s signature before calling it.

        Without this, a wrong/missing argument from the model surfaces as a
        raw Python ``TypeError`` (e.g. "func() missing 1 required positional
        argument: 'age'") mixed in with genuine runtime errors from inside
        the tool body, with no indication of what the correct schema was.
        Binding first lets us return a message that names the expected
        signature so the model can see what to fix, and never actually
        calls the tool with args that would fail anyway.
        """
        try:
            inspect.signature(func).bind(**tool_input)
        except TypeError as exc:
            try:
                sig = str(inspect.signature(func))
            except (TypeError, ValueError):
                sig = "(...)"
            return (
                f"Error: invalid arguments for tool {getattr(func, '__name__', 'tool')!r} — {exc}. "
                f"Expected signature: {getattr(func, '__name__', 'tool')}{sig}. Received: {tool_input!r}"
            )
        return None

    def _execute_tool(self, tool_map: Dict[str, Any], tool_name: str, tool_input: Any) -> str:
        """Call a tool function and return its string result."""
        max_chars: int = getattr(self, "MAX_TOOL_OUTPUT_CHARS", 5000)

        if tool_name not in tool_map:
            available = list(tool_map.keys())
            return f"Error: tool '{tool_name}' not found. Available tools: {available}"

        tool = tool_map[tool_name]
        func = _tool_func(tool)

        if func is None:
            return f"Error: tool '{tool_name}' has no callable 'func' key."

        if isinstance(tool_input, dict):
            validation_error = self._validate_tool_args(func, tool_input)
            if validation_error is not None:
                return validation_error

        try:
            if isinstance(tool_input, dict):
                result = str(func(**tool_input))
            else:
                result = str(func(tool_input))
        except Exception as exc:
            cb: CallbackManager = getattr(self, "callback_manager", None)
            if cb:
                cb.fire_tool_error(tool_name, exc, agent=self)
            result = f"Error executing '{tool_name}': {exc}"

        if len(result) > max_chars:
            result = result[:max_chars] + "... [truncated]"

        return result

    async def _execute_tool_async(self, tool_map: Dict[str, Any], tool_name: str, tool_input: Any) -> str:
        """Async version of _execute_tool — awaits coroutine funcs if needed."""
        max_chars: int = getattr(self, "MAX_TOOL_OUTPUT_CHARS", 5000)

        if tool_name not in tool_map:
            return f"Error: tool '{tool_name}' not found. Available: {list(tool_map.keys())}"

        tool = tool_map[tool_name]
        func = _tool_func(tool)

        if func is None:
            return f"Error: tool '{tool_name}' has no callable 'func' key."

        if isinstance(tool_input, dict):
            validation_error = self._validate_tool_args(func, tool_input)
            if validation_error is not None:
                return validation_error

        try:
            if isinstance(tool_input, dict):
                raw_result = func(**tool_input)
            else:
                raw_result = func(tool_input)

            if inspect.isawaitable(raw_result):
                raw_result = await raw_result

            result = str(raw_result)
        except Exception as exc:
            cb: CallbackManager = getattr(self, "callback_manager", None)
            if cb:
                cb.fire_tool_error(tool_name, exc, agent=self)
            result = f"Error executing '{tool_name}': {exc}"

        if len(result) > max_chars:
            result = result[:max_chars] + "... [truncated]"

        return result

    def _call_llm_with_retry(self, fn: Callable[[], Any]) -> Any:
        """Call a zero-arg LLM invocation (``lambda: self.llm.invoke(...)``),
        retrying on failure per llm_retries/llm_retry_backoff/llm_retry_on.

        A transient failure (rate limit, network blip) used to surface
        straight to AgentLLMError and end the run on the very first bad
        response, even though the same call would likely succeed a moment
        later. Retries with exponential backoff (base * 2**attempt, capped
        at llm_retry_max_backoff) give it that moment. llm_retries=0 (the
        default) makes this a single unconditional call, identical to prior
        behavior.
        """
        retries: int = getattr(self, "llm_retries", 0) or 0
        backoff: float = getattr(self, "llm_retry_backoff", 1.0)
        max_backoff: float = getattr(self, "llm_retry_max_backoff", 30.0)
        should_retry: Callable[[BaseException], bool] = getattr(self, "llm_retry_on", None) or _default_should_retry
        logger = getattr(self, "logger", None)

        attempt = 0
        while True:
            try:
                return fn()
            except Exception as exc:
                if attempt >= retries or not should_retry(exc):
                    raise
                delay = min(backoff * (2 ** attempt), max_backoff)
                if logger:
                    logger.info(
                        f"LLM call failed ({exc}); retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{retries})."
                    )
                time.sleep(delay)
                attempt += 1

    async def _acall_llm_with_retry(self, coro_fn: Callable[[], Any]) -> Any:
        """Async twin of _call_llm_with_retry -- ``coro_fn`` is a zero-arg
        callable returning an awaitable (``lambda: self.llm.ainvoke(...)``),
        awaited fresh on each attempt since a coroutine object can't be
        awaited twice."""
        retries: int = getattr(self, "llm_retries", 0) or 0
        backoff: float = getattr(self, "llm_retry_backoff", 1.0)
        max_backoff: float = getattr(self, "llm_retry_max_backoff", 30.0)
        should_retry: Callable[[BaseException], bool] = getattr(self, "llm_retry_on", None) or _default_should_retry
        logger = getattr(self, "logger", None)

        attempt = 0
        while True:
            try:
                return await coro_fn()
            except Exception as exc:
                if attempt >= retries or not should_retry(exc):
                    raise
                delay = min(backoff * (2 ** attempt), max_backoff)
                if logger:
                    logger.info(
                        f"LLM call failed ({exc}); retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{retries})."
                    )
                await asyncio.sleep(delay)
                attempt += 1

    def _collect_future_result(self, tool_name: str, future: Any, timeout: Optional[float]) -> str:
        """Block on a submitted tool future, enforcing ``tool_timeout``.

        Without a timeout here, a hung tool call (e.g. a network request
        with no timeout of its own) blocks ``future.result()`` forever --
        ``max_execution_time`` can't save you, since it's only ever checked
        at the *start* of an iteration, not while a tool call is in flight.
        A timed-out future's underlying thread is NOT killed (Python has no
        way to force-stop a running thread) -- it keeps running in the
        background until it finishes or the process exits -- but the agent
        loop itself is no longer blocked on it.
        """
        try:
            return future.result(timeout=timeout)
        except _FutureTimeoutError:
            result = f"Error: tool '{tool_name}' timed out after {timeout}s."
            cb: CallbackManager = getattr(self, "callback_manager", None)
            if cb:
                cb.fire_tool_error(tool_name, TimeoutError(result), agent=self)
            return result

    async def _execute_tool_async_with_timeout(
        self, tool_map: Dict[str, Any], tool_name: str, tool_input: Any, timeout: Optional[float]
    ) -> str:
        """Async twin of _collect_future_result -- wraps _execute_tool_async in
        asyncio.wait_for so a hung async tool doesn't block the loop forever.
        For a genuinely async tool func, wait_for can actually cancel it at
        its next await point; for a blocking sync tool func, this can only
        raise once the underlying call returns (asyncio can't preempt a
        running sync frame), same fundamental limit as anywhere else a sync
        function runs inside an event loop.
        """
        try:
            return await asyncio.wait_for(
                self._execute_tool_async(tool_map, tool_name, tool_input), timeout=timeout
            )
        except asyncio.TimeoutError:
            result = f"Error: tool '{tool_name}' timed out after {timeout}s."
            cb: CallbackManager = getattr(self, "callback_manager", None)
            if cb:
                cb.fire_tool_error(tool_name, TimeoutError(result), agent=self)
            return result

    def _trim_scratchpad(self, scratchpad: str) -> str:
        """Trim the scratchpad to fit both the character cap (MAX_SCRATCHPAD_CHARS,
        always active) and, if set, a token budget (max_scratchpad_tokens) --
        char count alone is a poor proxy for what actually overflows an LLM's
        context window, since tokens-per-char varies a lot by language and
        content (dense non-English text or code can run well under 4
        chars/token, silently blowing a char-only budget's whole point).
        """
        max_chars: int = getattr(self, "MAX_SCRATCHPAD_CHARS", 15000)
        if len(scratchpad) > max_chars:
            scratchpad = "[...earlier steps trimmed...]\n" + scratchpad[-max_chars:]

        max_tokens: Optional[int] = getattr(self, "max_scratchpad_tokens", None)
        if max_tokens is not None:
            counter: Callable[[str], int] = getattr(self, "token_counter", None) or _default_token_counter
            if counter(scratchpad) > max_tokens:
                scratchpad = _trim_to_token_budget(scratchpad, max_tokens, counter)

        return scratchpad

    # ── sync loop ─────────────────────────────────────────────────────────────

    def _run_loop(
        self,
        query: str,
        max_iterations: int,
        approval_callback: Optional[Callable[[str, Dict[str, Any]], Any]],
        extra_kwargs: Dict[str, Any],
    ) -> str:
        self.scratchpad = ""
        consecutive_parse_errors = 0
        start_time = time.monotonic()
        max_parse_errors: int = getattr(self, "max_consecutive_parse_errors",
                                        getattr(self, "MAX_CONSECUTIVE_PARSE_ERRORS", 3))
        max_exec_time: Optional[float] = getattr(self, "max_execution_time", None)
        tool_timeout: Optional[float] = getattr(self, "tool_timeout", None)
        template: str = getattr(self, "prompt_template", "")
        logger = getattr(self, "logger", None)
        cb: CallbackManager = getattr(self, "callback_manager", CallbackManager())
        memory_context = _get_memory_context(getattr(self, "memory", None), query)

        for iteration in range(1, max_iterations + 1):
            cb.fire_iteration_start(iteration, agent=self)
            iteration_extra_kwargs = cb.fire_before_iteration(iteration, agent=self)

            # time guard
            if max_exec_time and (time.monotonic() - start_time) > max_exec_time:
                raise AgentTimeoutError(max_exec_time)

            # Rebuild from self.tools every iteration (not once before the loop)
            # so a tool added mid-run (e.g. autourgos-toolbox's expose_toolbox())
            # is both advertised to the LLM and actually callable on the very
            # next iteration -- a stale snapshot here silently made every such
            # middleware's "expose more tools mid-run" feature never work.
            current_tools = getattr(self, "tools", [])
            tool_map: Dict[str, Any] = {_tool_name(t): t for t in current_tools}

            # render prompt
            prompt_text = template.format(
                tool_list=build_tool_list(current_tools),
                previous_context=self.scratchpad or "None",
                user_input=query,
                memory_context=memory_context,
            )
            messages = self._build_messages(prompt_text)

            # call LLM
            call_kwargs = {**extra_kwargs, **iteration_extra_kwargs}
            try:
                raw = self._call_llm_with_retry(lambda: self.llm.invoke(messages, **call_kwargs))  # type: ignore[attr-defined]
                response_text = self._extract_text(raw)
            except Exception as exc:
                raise AgentLLMError(exc) from exc

            cb.fire_llm_end(response_text, agent=self, raw=raw, **self._extract_llm_metadata(raw))

            if logger and getattr(logger, "full_output", False):
                logger.llm_response(response_text, iteration)

            # parse
            try:
                thought, actions, final_answer = self._parser(response_text)  # type: ignore[attr-defined]
            except Exception:
                thought, actions, final_answer = None, [], None

            if thought:
                cb.fire_iteration(iteration, thought, agent=self)
                if logger:
                    logger.thought(thought, iteration)

            # final answer
            if final_answer:
                memory = getattr(self, "memory", None)
                if memory:
                    _record_agent_message(memory, final_answer)
                cb.fire_agent_end(final_answer, agent=self)
                if logger:
                    logger.final_answer(final_answer)
                return final_answer

            # no actions — parse error
            if not actions:
                # NOTE: this counter must only reset once a turn actually
                # produces actions/final_answer (below) -- resetting it
                # unconditionally on every non-throwing _parser() call (as
                # this used to) meant a response like "not json" (which
                # parse_json_object tolerates by returning {} rather than
                # raising) reset the counter right back to 0 every single
                # iteration, so it could never reach max_parse_errors and
                # AgentParseError could never actually fire.
                consecutive_parse_errors += 1
                cb.fire_parse_error(iteration, response_text, agent=self)
                if logger:
                    logger.parse_error(response_text, iteration)
                if consecutive_parse_errors >= max_parse_errors:
                    raise AgentParseError(consecutive_parse_errors, response_text)
                self.scratchpad += (
                    f"\nStep {iteration}:\n"
                    f"Thought: {thought or 'None'}\n"
                    f"Observation: Response was not valid JSON. Please reply with the exact JSON format.\n"
                )
                self.scratchpad = self._trim_scratchpad(self.scratchpad)
                continue

            consecutive_parse_errors = 0

            # execute tools — the prompt tells the model it can request several
            # independent tool calls in one turn ("You can call multiple tools
            # at once if they don't depend on each other's outputs"), so the
            # approved ones actually run concurrently here to honor that,
            # instead of one at a time.
            step_lines: List[str] = [f"\nStep {iteration}:"]
            if thought:
                step_lines.append(f"Thought: {thought}")

            approved: List[Tuple[str, Any]] = []
            for action_dict in actions:
                tool_name  = action_dict.get("action", "")
                tool_input = action_dict.get("action_input", {})

                if logger:
                    logger.tool_call(tool_name, tool_input, iteration)
                cb.fire_tool_start(tool_name, tool_input, agent=self)

                # approval gate
                if approval_callback and not _call_sync_approval(approval_callback, tool_name, tool_input):
                    denial_result = "Tool call was denied by the approval callback."
                    cb.fire_tool_end(tool_name, denial_result, agent=self)
                    step_lines.append(f"Action: {tool_name}({tool_input})")
                    step_lines.append(f"Observation: {denial_result}")
                    continue

                approved.append((tool_name, tool_input))

            if approved:
                max_workers = min(len(approved), getattr(self, "MAX_TOOL_WORKERS", 8))
                # Not a `with` block: ThreadPoolExecutor.__exit__ calls
                # shutdown(wait=True), which blocks until every submitted
                # thread finishes -- including one _collect_future_result
                # already gave up on via tool_timeout. shutdown(wait=False)
                # lets the loop move on immediately; the timed-out thread
                # (if any) is abandoned to finish on its own, same as
                # documented on _collect_future_result.
                pool = ThreadPoolExecutor(max_workers=max_workers)
                try:
                    futures = [
                        (tool_name, tool_input, pool.submit(self._execute_tool, tool_map, tool_name, tool_input))
                        for tool_name, tool_input in approved
                    ]
                    for tool_name, tool_input, future in futures:
                        result = self._collect_future_result(tool_name, future, tool_timeout)
                        cb.fire_tool_end(tool_name, result, agent=self)
                        if logger:
                            logger.tool_result(tool_name, result, iteration)
                        step_lines.append(f"Action: {tool_name}({tool_input})")
                        step_lines.append(f"Observation: {result}")
                finally:
                    pool.shutdown(wait=False)

            self.scratchpad += "\n".join(step_lines)
            self.scratchpad = self._trim_scratchpad(self.scratchpad)

        raise AgentMaxIterationsError(max_iterations)

    # ── async loop ────────────────────────────────────────────────────────────

    async def _arun_loop(
        self,
        query: str,
        max_iterations: int,
        approval_callback: Optional[Callable[[str, Dict[str, Any]], Any]],
        extra_kwargs: Dict[str, Any],
    ) -> str:
        self.scratchpad = ""
        consecutive_parse_errors = 0
        start_time = time.monotonic()
        max_parse_errors: int = getattr(self, "max_consecutive_parse_errors",
                                        getattr(self, "MAX_CONSECUTIVE_PARSE_ERRORS", 3))
        max_exec_time: Optional[float] = getattr(self, "max_execution_time", None)
        tool_timeout: Optional[float] = getattr(self, "tool_timeout", None)
        template: str = getattr(self, "prompt_template", "")
        logger = getattr(self, "logger", None)
        cb: CallbackManager = getattr(self, "callback_manager", CallbackManager())
        memory_context = _get_memory_context(getattr(self, "memory", None), query)

        for iteration in range(1, max_iterations + 1):
            cb.fire_iteration_start(iteration, agent=self)
            iteration_extra_kwargs = cb.fire_before_iteration(iteration, agent=self)

            if max_exec_time and (time.monotonic() - start_time) > max_exec_time:
                raise AgentTimeoutError(max_exec_time)

            # See _run_loop's identical comment: rebuilt every iteration so a
            # tool added mid-run is actually callable on the next iteration.
            current_tools = getattr(self, "tools", [])
            tool_map: Dict[str, Any] = {_tool_name(t): t for t in current_tools}

            prompt_text = template.format(
                tool_list=build_tool_list(current_tools),
                previous_context=self.scratchpad or "None",
                user_input=query,
                memory_context=memory_context,
            )
            messages = self._build_messages(prompt_text)

            call_kwargs = {**extra_kwargs, **iteration_extra_kwargs}
            try:
                raw = await self._acall_llm_with_retry(lambda: self.llm.ainvoke(messages, **call_kwargs))  # type: ignore[attr-defined]
                response_text = self._extract_text(raw)
            except Exception as exc:
                raise AgentLLMError(exc) from exc

            cb.fire_llm_end(response_text, agent=self, raw=raw, **self._extract_llm_metadata(raw))

            if logger and getattr(logger, "full_output", False):
                logger.llm_response(response_text, iteration)

            try:
                thought, actions, final_answer = self._parser(response_text)  # type: ignore[attr-defined]
            except Exception:
                thought, actions, final_answer = None, [], None

            if thought:
                cb.fire_iteration(iteration, thought, agent=self)
                if logger:
                    logger.thought(thought, iteration)

            if final_answer:
                memory = getattr(self, "memory", None)
                if memory:
                    _record_agent_message(memory, final_answer)
                cb.fire_agent_end(final_answer, agent=self)
                if logger:
                    logger.final_answer(final_answer)
                return final_answer

            if not actions:
                # see _run_loop's identical comment: must only reset once a
                # turn actually produces actions/final_answer, not on every
                # non-throwing _parser() call, or this can never trip.
                consecutive_parse_errors += 1
                cb.fire_parse_error(iteration, response_text, agent=self)
                if logger:
                    logger.parse_error(response_text, iteration)
                if consecutive_parse_errors >= max_parse_errors:
                    raise AgentParseError(consecutive_parse_errors, response_text)
                self.scratchpad += (
                    f"\nStep {iteration}:\n"
                    f"Thought: {thought or 'None'}\n"
                    f"Observation: Response was not valid JSON. Please reply with the exact JSON format.\n"
                )
                self.scratchpad = self._trim_scratchpad(self.scratchpad)
                continue

            consecutive_parse_errors = 0

            step_lines: List[str] = [f"\nStep {iteration}:"]
            if thought:
                step_lines.append(f"Thought: {thought}")

            approved: List[Tuple[str, Any]] = []
            for action_dict in actions:
                tool_name  = action_dict.get("action", "")
                tool_input = action_dict.get("action_input", {})

                if logger:
                    logger.tool_call(tool_name, tool_input, iteration)
                cb.fire_tool_start(tool_name, tool_input, agent=self)

                if approval_callback:
                    is_approved = await _maybe_await(approval_callback(tool_name, tool_input))
                    if not is_approved:
                        denial_result = "Tool call was denied by the approval callback."
                        cb.fire_tool_end(tool_name, denial_result, agent=self)
                        step_lines.append(f"Action: {tool_name}({tool_input})")
                        step_lines.append(f"Observation: {denial_result}")
                        continue

                approved.append((tool_name, tool_input))

            if approved:
                results = await asyncio.gather(*[
                    self._execute_tool_async_with_timeout(tool_map, tool_name, tool_input, tool_timeout)
                    for tool_name, tool_input in approved
                ])
                for (tool_name, tool_input), result in zip(approved, results):
                    cb.fire_tool_end(tool_name, result, agent=self)
                    if logger:
                        logger.tool_result(tool_name, result, iteration)
                    step_lines.append(f"Action: {tool_name}({tool_input})")
                    step_lines.append(f"Observation: {result}")

            self.scratchpad += "\n".join(step_lines)
            self.scratchpad = self._trim_scratchpad(self.scratchpad)

        raise AgentMaxIterationsError(max_iterations)

    # ── native tool-calling loop (tool_calling_mode="native") ────────────────
    # Uses the LLM's own invoke_with_tools()/ainvoke_with_tools() -- structured
    # tool_calls straight from the API -- instead of hand-rolled JSON-in-text
    # parsing. Conversation state is a real multi-turn message list, not a
    # single rendered scratchpad string; self.scratchpad is still kept up to
    # date (human-readable trace only, not fed back to the LLM) so middleware
    # relying on the scratchpad contract still sees something sensible.

    def _build_native_messages(self, query: str, memory_context: str = "") -> List[Dict[str, Any]]:
        system_prompt: str = getattr(self, "system_prompt", "")
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if memory_context:
            messages.append({"role": "system", "content": memory_context})
        messages.append({"role": "user", "content": query})
        return messages

    def _wrap_unsupported_native_error(self, exc: NotImplementedError) -> RuntimeError:
        llm_class = type(getattr(self, "llm", None)).__name__
        return RuntimeError(
            f"tool_calling_mode='native' requires {llm_class} to implement "
            f"invoke_with_tools()/ainvoke_with_tools(), but it raised "
            f"NotImplementedError: {exc}. Use tool_calling_mode='prompt' "
            f"(the default) with this LLM instead."
        )

    def _require_native_tool_calling_support(self, method_name: str) -> None:
        """
        Fail fast, before the loop starts, if the LLM has no such method at
        all -- e.g. a custom duck-typed LLM, or one built on a BaseLLM that
        doesn't declare invoke_with_tools()/ainvoke_with_tools() as an
        overridable stub. This is distinct from the NotImplementedError case
        handled around the actual call below (a BaseLLM that DOES declare
        the method, but the concrete subclass hasn't implemented it) --
        without this check, a missing attribute would raise a plain
        AttributeError mid-loop and get swallowed into a generic
        "[LLM Error] ..." string instead of a clear, actionable message.
        """
        llm = getattr(self, "llm", None)
        method = getattr(llm, method_name, None)
        if method is None or not callable(method):
            raise RuntimeError(
                f"tool_calling_mode='native' requires {type(llm).__name__} to "
                f"implement {method_name}(), but it has no such attribute. Use "
                f"tool_calling_mode='prompt' (the default) with this LLM instead."
            )

    def _record_native_step(self, iteration: int, calls_and_results: List[Tuple[Any, str]]) -> None:
        step_lines: List[str] = [f"\nStep {iteration}:"]
        for tc, result in calls_and_results:
            step_lines.append(f"Action: {tc.name}({tc.arguments})")
            step_lines.append(f"Observation: {result}")
        self.scratchpad += "\n".join(step_lines)
        self.scratchpad = self._trim_scratchpad(self.scratchpad)

    @staticmethod
    def _assistant_tool_call_message(tool_calls: List[Any]) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in tool_calls
            ],
        }

    def _gate_tool_calls_for_approval(
        self,
        tool_calls: List[Any],
        approval_callback: Optional[Callable[[str, Dict[str, Any]], Any]],
        cb: "CallbackManager",
        logger: Any,
        iteration: int,
    ) -> Tuple[List[Any], List[Tuple[Any, str]]]:
        """Split tool_calls into (approved, [(call, result) for denied ones])."""
        approved: List[Any] = []
        denied: List[Tuple[Any, str]] = []
        for tc in tool_calls:
            if logger:
                logger.tool_call(tc.name, tc.arguments, iteration)
            cb.fire_tool_start(tc.name, tc.arguments, agent=self)
            if approval_callback and not _call_sync_approval(approval_callback, tc.name, tc.arguments):
                result = "Tool call was denied by the approval callback."
                cb.fire_tool_end(tc.name, result, agent=self)
                denied.append((tc, result))
            else:
                approved.append(tc)
        return approved, denied

    async def _gate_tool_calls_for_approval_async(
        self,
        tool_calls: List[Any],
        approval_callback: Optional[Callable[[str, Dict[str, Any]], Any]],
        cb: "CallbackManager",
        logger: Any,
        iteration: int,
    ) -> Tuple[List[Any], List[Tuple[Any, str]]]:
        """Async twin of _gate_tool_calls_for_approval — awaits approval_callback
        if it returns an awaitable, instead of calling it synchronously from
        inside an async loop."""
        approved: List[Any] = []
        denied: List[Tuple[Any, str]] = []
        for tc in tool_calls:
            if logger:
                logger.tool_call(tc.name, tc.arguments, iteration)
            cb.fire_tool_start(tc.name, tc.arguments, agent=self)
            if approval_callback and not await _maybe_await(approval_callback(tc.name, tc.arguments)):
                result = "Tool call was denied by the approval callback."
                cb.fire_tool_end(tc.name, result, agent=self)
                denied.append((tc, result))
            else:
                approved.append(tc)
        return approved, denied

    def _run_loop_native(
        self,
        query: str,
        max_iterations: int,
        approval_callback: Optional[Callable[[str, Dict[str, Any]], Any]],
        extra_kwargs: Dict[str, Any],
    ) -> str:
        self._require_native_tool_calling_support("invoke_with_tools")
        self.scratchpad = ""
        consecutive_empty = 0
        start_time = time.monotonic()
        max_empty: int = getattr(self, "max_consecutive_parse_errors",
                                  getattr(self, "MAX_CONSECUTIVE_PARSE_ERRORS", 3))
        max_exec_time: Optional[float] = getattr(self, "max_execution_time", None)
        tool_timeout: Optional[float] = getattr(self, "tool_timeout", None)
        logger = getattr(self, "logger", None)
        cb: CallbackManager = getattr(self, "callback_manager", CallbackManager())
        messages = self._build_native_messages(query, _get_memory_context(getattr(self, "memory", None), query))

        for iteration in range(1, max_iterations + 1):
            cb.fire_iteration_start(iteration, agent=self)
            iteration_extra_kwargs = cb.fire_before_iteration(iteration, agent=self)

            if max_exec_time and (time.monotonic() - start_time) > max_exec_time:
                raise AgentTimeoutError(max_exec_time)

            # Rebuilt every iteration -- see _run_loop's identical comment.
            # self.tools is already passed live to invoke_with_tools() below,
            # but tool_map (used to actually execute an approved call further
            # down) was previously frozen once before the loop, so a tool
            # exposed mid-run could be advertised to the LLM yet still fail
            # with "not found" the moment it tried to call it.
            tool_map: Dict[str, Any] = {_tool_name(t): t for t in getattr(self, "tools", [])}

            call_kwargs = {**extra_kwargs, **iteration_extra_kwargs}
            try:
                response = self._call_llm_with_retry(
                    lambda: self.llm.invoke_with_tools(messages, self.tools, **call_kwargs)  # type: ignore[attr-defined]
                )
            except NotImplementedError as exc:
                raise self._wrap_unsupported_native_error(exc) from exc
            except Exception as exc:
                raise AgentLLMError(exc) from exc

            cb.fire_llm_end(
                response.text if response.is_final_answer else None,
                agent=self,
                raw=response.raw,
                **self._extract_llm_metadata(response.raw),
            )

            if response.is_final_answer:
                final_answer = response.text
                memory = getattr(self, "memory", None)
                if memory:
                    _record_agent_message(memory, final_answer)
                cb.fire_agent_end(final_answer, agent=self)
                if logger:
                    logger.final_answer(final_answer)
                return final_answer

            if not response.has_tool_calls:
                consecutive_empty += 1
                if logger:
                    logger.parse_error("<empty response: no text and no tool_calls>", iteration)
                if consecutive_empty >= max_empty:
                    raise AgentEmptyResponseError(consecutive_empty)
                messages.append({"role": "user", "content": "Please either call a tool or give a final answer."})
                continue
            consecutive_empty = 0

            messages.append(self._assistant_tool_call_message(response.tool_calls))
            approved, calls_and_results = self._gate_tool_calls_for_approval(
                response.tool_calls, approval_callback, cb, logger, iteration
            )

            if approved:
                max_workers = min(len(approved), getattr(self, "MAX_TOOL_WORKERS", 8))
                # Not a `with` block -- see the identical note in _run_loop:
                # shutdown(wait=True) on __exit__ would block on a thread
                # _collect_future_result already gave up on via tool_timeout.
                pool = ThreadPoolExecutor(max_workers=max_workers)
                try:
                    futures = [(tc, pool.submit(self._execute_tool, tool_map, tc.name, tc.arguments)) for tc in approved]
                    for tc, future in futures:
                        result = self._collect_future_result(tc.name, future, tool_timeout)
                        cb.fire_tool_end(tc.name, result, agent=self)
                        if logger:
                            logger.tool_result(tc.name, result, iteration)
                        calls_and_results.append((tc, result))
                finally:
                    pool.shutdown(wait=False)

            results_by_call_id = {tc.call_id: result for tc, result in calls_and_results}
            for tc in response.tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": results_by_call_id.get(tc.call_id, "")})

            self._record_native_step(iteration, calls_and_results)

        raise AgentMaxIterationsError(max_iterations)

    async def _arun_loop_native(
        self,
        query: str,
        max_iterations: int,
        approval_callback: Optional[Callable[[str, Dict[str, Any]], Any]],
        extra_kwargs: Dict[str, Any],
    ) -> str:
        self._require_native_tool_calling_support("ainvoke_with_tools")
        self.scratchpad = ""
        consecutive_empty = 0
        start_time = time.monotonic()
        max_empty: int = getattr(self, "max_consecutive_parse_errors",
                                  getattr(self, "MAX_CONSECUTIVE_PARSE_ERRORS", 3))
        max_exec_time: Optional[float] = getattr(self, "max_execution_time", None)
        tool_timeout: Optional[float] = getattr(self, "tool_timeout", None)
        logger = getattr(self, "logger", None)
        cb: CallbackManager = getattr(self, "callback_manager", CallbackManager())
        messages = self._build_native_messages(query, _get_memory_context(getattr(self, "memory", None), query))

        for iteration in range(1, max_iterations + 1):
            cb.fire_iteration_start(iteration, agent=self)
            iteration_extra_kwargs = cb.fire_before_iteration(iteration, agent=self)

            if max_exec_time and (time.monotonic() - start_time) > max_exec_time:
                raise AgentTimeoutError(max_exec_time)

            # Rebuilt every iteration -- see _run_loop_native's identical comment.
            tool_map: Dict[str, Any] = {_tool_name(t): t for t in getattr(self, "tools", [])}

            call_kwargs = {**extra_kwargs, **iteration_extra_kwargs}
            try:
                response = await self._acall_llm_with_retry(
                    lambda: self.llm.ainvoke_with_tools(messages, self.tools, **call_kwargs)  # type: ignore[attr-defined]
                )
            except NotImplementedError as exc:
                raise self._wrap_unsupported_native_error(exc) from exc
            except Exception as exc:
                raise AgentLLMError(exc) from exc

            cb.fire_llm_end(
                response.text if response.is_final_answer else None,
                agent=self,
                raw=response.raw,
                **self._extract_llm_metadata(response.raw),
            )

            if response.is_final_answer:
                final_answer = response.text
                memory = getattr(self, "memory", None)
                if memory:
                    _record_agent_message(memory, final_answer)
                cb.fire_agent_end(final_answer, agent=self)
                if logger:
                    logger.final_answer(final_answer)
                return final_answer

            if not response.has_tool_calls:
                consecutive_empty += 1
                if logger:
                    logger.parse_error("<empty response: no text and no tool_calls>", iteration)
                if consecutive_empty >= max_empty:
                    raise AgentEmptyResponseError(consecutive_empty)
                messages.append({"role": "user", "content": "Please either call a tool or give a final answer."})
                continue
            consecutive_empty = 0

            messages.append(self._assistant_tool_call_message(response.tool_calls))
            approved, calls_and_results = await self._gate_tool_calls_for_approval_async(
                response.tool_calls, approval_callback, cb, logger, iteration
            )

            if approved:
                results = await asyncio.gather(*[
                    self._execute_tool_async_with_timeout(tool_map, tc.name, tc.arguments, tool_timeout)
                    for tc in approved
                ])
                for tc, result in zip(approved, results):
                    cb.fire_tool_end(tc.name, result, agent=self)
                    if logger:
                        logger.tool_result(tc.name, result, iteration)
                    calls_and_results.append((tc, result))

            results_by_call_id = {tc.call_id: result for tc, result in calls_and_results}
            for tc in response.tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": results_by_call_id.get(tc.call_id, "")})

            self._record_native_step(iteration, calls_and_results)

        raise AgentMaxIterationsError(max_iterations)


# ── BaseAgent ──────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base class for all Autourgos agents.

    Manages tools, memory, and the callback manager.
    Concrete agents extend this and implement invoke() / ainvoke().
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        memory: Optional[MemoryProtocol] = None,
        verbose: bool = False,
        max_iterations: int = 15,
        max_execution_time: Optional[float] = None,
        middleware: Optional[List[CallbackHandler]] = None,
        tools: Optional[List[Any]] = None,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time
        self.callback_manager = CallbackManager(middleware or [])
        self.tools: List[Any] = list(tools or [])

        # Middleware integration contract: live, externally-readable state.
        # Both are reset to their initial values at the start of every
        # invoke()/ainvoke() call, and updated in place as the loop runs,
        # so a callback handler (or any external code holding a reference
        # to the agent) can read them mid-run.
        self.scratchpad: str = ""
        self.current_query: str = ""

    def add_tools(self, *tools: Any) -> "BaseAgent":
        """
        Add one or more tools to the agent.

        Accepts individual tool dicts or lists of tool dicts::

            agent.add_tools(tool_a, tool_b)
            agent.add_tools([tool_a, tool_b])

        A tool whose name collides with an already-registered tool replaces
        it (so intentional overrides keep working) rather than being kept
        alongside it: the loop's tool_map is built as {name: tool}, so only
        the most recently added implementation with that name would ever
        execute anyway — keeping both around just meant the LLM saw the
        same name listed twice, with two different (and often contradictory)
        descriptions, for a tool that only ever ran one way. A warning is
        still logged so silent overrides don't go unnoticed.
        """
        for item in tools:
            for t in (item if isinstance(item, list) else [item]):
                name = _tool_name(t)
                if name is not None:
                    for existing in list(self.tools):
                        if _tool_name(existing) == name:
                            _logger.warning(
                                "Tool name %r is already registered on this agent — "
                                "replacing the earlier tool with this new one. Rename "
                                "one of them if this wasn't intentional.",
                                name,
                            )
                            self.tools.remove(existing)
                self.tools.append(t)
        return self

    def add_middleware(self, handler: CallbackHandler) -> "BaseAgent":
        """Register a CallbackHandler for lifecycle events."""
        self.callback_manager.add_handler(handler)
        return self

    def reset_tools(self) -> "BaseAgent":
        """Remove all tools."""
        self.tools = []
        return self

    @abstractmethod
    def invoke(self, query: str, **kwargs: Any) -> str:
        ...

    @abstractmethod
    async def ainvoke(self, query: str, **kwargs: Any) -> str:
        ...

    def __enter__(self) -> "BaseAgent":
        return self

    def __exit__(self, *_: Any) -> None:
        llm = getattr(self, "llm", None)
        if llm and hasattr(llm, "close"):
            try:
                llm.close()
            except Exception:
                _logger.warning("llm.close() raised during agent cleanup", exc_info=True)

    async def __aenter__(self) -> "BaseAgent":
        return self

    async def __aexit__(self, *_: Any) -> None:
        llm = getattr(self, "llm", None)
        if llm and hasattr(llm, "aclose"):
            try:
                await llm.aclose()
            except Exception:
                _logger.warning("llm.aclose() raised during agent cleanup", exc_info=True)
