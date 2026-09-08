"""
base.py — Self-contained base classes for autourgos-agent.

Inlines BaseLLM, BaseAgent, AgentLoopMixin, CallbackManager, and all
protocols. Depends only on autourgos-core (a separate, zero-dependency
stdlib utility library shared across the framework) -- no other
third-party or autourgos-* dependency.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeoutError
from typing import Any, Callable, Dict, List, Optional, Tuple

from autourgos_core import aretry_with_backoff, extract_text as _extract_text_fn, retry_with_backoff

from .history import _NULL_HISTORY
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


class AgentAlreadyRunningError(AgentError):
    """Raised when invoke()/ainvoke() is called on an Agent instance that
    already has a run in progress. An Agent's mid-run state (scratchpad,
    current_query) is shared, mutable instance state read by middleware
    during the run -- a second concurrent call on the same instance would
    silently overwrite it out from under the first. Use a separate Agent
    instance for concurrent work instead."""

    def __init__(self) -> None:
        super().__init__(
            "This Agent instance already has a run in progress. "
            "invoke()/ainvoke() cannot be called concurrently on the same "
            "instance -- use a separate Agent instance for concurrent work."
        )


# ── Protocols ─────────────────────────────────────────────────────────────────

class CallbackHandler:
    """
    Base class for agent middleware / event hooks.

    Sub-class and override the methods you care about.  Unused methods
    are no-ops so you never have to implement every hook.

    13 hooks total:
      - on_agent_start, on_agent_end, on_agent_error
      - on_agent_pause, on_agent_resume
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

    def on_agent_pause(self, iteration: int, reason: Optional[str], agent: Any = None, **kwargs: Any) -> None:
        """Called once, right before the run blocks at an iteration boundary
        because ``Agent.pause()`` was called (and ``resume()`` hasn't yet).
        ``reason`` is whatever string was passed to ``pause(reason=...)``,
        or None. See README's "Pause & Resume" section."""
        pass

    def on_agent_resume(self, iteration: int, paused_duration: float, agent: Any = None, **kwargs: Any) -> None:
        """Called once, right after the run unblocks because ``Agent.resume()``
        was called. ``paused_duration`` is how long (in seconds) this
        particular pause lasted."""
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

    def on_iteration(
        self, iteration: int, thought: Optional[str], agent: Any = None, **kwargs: Any
    ) -> Optional[bool]:
        """
        Called once per prompt-mode loop iteration, right after a thought
        (if any) is parsed from the LLM response and before that response's
        action batch is dispatched. NOT called in tool_calling_mode="native"
        (no equivalent point exists there today).

        Return a truthy value to discard this iteration's action batch
        instead of executing it -- e.g. because this call injected a newer
        human instruction (HcixInterruptMiddleware) and the reasoning
        behind those actions is now stale. The loop appends a scratchpad
        note and moves straight to the next iteration's LLM call, which
        will see whatever this call injected. When multiple handlers are
        registered, any single truthy return discards the batch (OR
        semantics) -- there's no "un-discard" once one handler asks for it.
        Return None/falsy (the default) for no-op, matching prior behavior
        where this hook's return value was purely informational.
        """
        return None

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
    """Duck-typed tool name lookup: plain dicts and Tool (dict subclass)
    alike, falling back to a bare callable's __name__ if it has neither a
    "name" key nor a .name attribute -- needed by consumers (e.g.
    autourgos-toolbox's Toolbox) whose tool lists can hold raw, unnormalized
    callables, not just this package's own always-normalized dict shape."""
    if isinstance(tool, dict):
        return tool.get("name")
    name = getattr(tool, "name", None)
    if name is not None:
        return name
    return getattr(tool, "__name__", None) if callable(tool) else None


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


def _trim_to_token_budget(
    text: str, max_tokens: int, counter: Callable[[str], int],
    prefix: str = "[...earlier steps trimmed...]\n",
) -> str:
    """Binary-search the longest tail of `text` whose token count (per
    `counter`) fits in `max_tokens` once `prefix` is accounted for, and
    return prefix + that tail.

    Binary search (not a linear scan) because `counter` may be a real
    tokenizer call, not just len() -- O(log n) calls keeps this cheap even
    for a large scratchpad. Token count isn't guaranteed strictly
    monotonic in string length for every possible tokenizer (a cut can
    occasionally merge/split a token differently), so this can be off by a
    token or two at the boundary -- acceptable for a soft trim guard, and
    the search still converges since it's monotonic enough in practice for
    real tokenizers and for the default char-based approximation.
    """
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
    """Fires lifecycle events to all registered handlers.

    Hook methods (``on_iteration_start``, ``on_tool_start``, etc.) may be
    defined as either a plain ``def`` or an ``async def`` on a
    ``CallbackHandler`` subclass -- both are supported from both the sync
    loop (``invoke()``) and the async loop (``ainvoke()``):

    - From the async loop, a sync hook is run in a background thread (via
      ``loop.run_in_executor``) so a blocking call inside it (an LLM
      request, a file write, anything) does not stall the event loop for
      the whole run; an async hook is awaited directly.
    - From the sync loop, an async hook is driven to completion with
      ``asyncio.run()`` (there's no event loop already running to await
      into); a sync hook is just called directly, as before.

    This mirrors ``_maybe_await``'s existing support for an async
    ``approval_callback``, applied to the rest of the middleware surface.
    """

    def __init__(self, handlers: Optional[List[CallbackHandler]] = None) -> None:
        self._handlers: List[CallbackHandler] = list(handlers or [])
        self._hook_executor: Optional[ThreadPoolExecutor] = None
        # Holds the *current async run's* contextvars.Context, so a sync
        # hook offloaded to _hook_executor (which does not run on the
        # event-loop thread, and therefore never sees ambient contextvar
        # writes on its own) can still read/write ContextVar-scoped
        # per-run state that another hook in the SAME run set earlier --
        # see capture_run_context()'s docstring for why this has to be one
        # reused Context object per run rather than a fresh copy per call.
        #
        # This is itself a ContextVar (not a plain attribute) specifically
        # so concurrent ainvoke() runs sharing this one CallbackManager
        # instance -- each typically its own asyncio Task, with its own
        # copied ambient context -- each see only their own run's Context
        # object here, never another concurrent run's.
        self._run_context_var: "contextvars.ContextVar[Optional[contextvars.Context]]" = (
            contextvars.ContextVar(f"autourgos_agent_run_context_{id(self)}", default=None)
        )

    def add_handler(self, handler: CallbackHandler) -> None:
        self._handlers.append(handler)

    def _get_hook_executor(self) -> ThreadPoolExecutor:
        if self._hook_executor is None:
            self._hook_executor = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="autourgos_agent_hooks"
            )
        return self._hook_executor

    def capture_run_context(self) -> None:
        """
        Snapshot the calling task's contextvars.Context and remember it as
        "this run's context" for every subsequent sync-hook offload to
        _hook_executor during this async run (_afire / afire_before_iteration).

        Call this once, at the very start of an async run (Agent.ainvoke()
        does this before firing on_agent_start), NOT before every individual
        hook call. contextvars.Context.run(callable) only makes a callable's
        ContextVar writes visible to a LATER Context.run() call on that exact
        same Context object -- a fresh contextvars.copy_context() per hook
        call (the naive fix) creates a new, unrelated snapshot each time, so
        an earlier hook's ContextVar.set() would never be visible to a later
        hook's read. Reusing one Context object for the whole run, retrieved
        via self._run_context_var.get() inside _afire, is what actually
        makes writes and reads made from different hook calls (even nested
        ones, e.g. concurrent tool-execution hooks under asyncio.gather --
        each gets its own Task whose context is a copy taken AFTER this
        Context object was already bound to _run_context_var, so the binding,
        a reference, carries through) see each other correctly.

        Hooks fired without ever calling this first (e.g. a caller invoking
        CallbackManager methods directly, outside Agent.ainvoke()) keep the
        prior behavior exactly: no context reuse, each sync-hook offload
        just runs on a bare worker thread as before.
        """
        self._run_context_var.set(contextvars.copy_context())

    @staticmethod
    def _accepts_agent_kwarg(fn: Callable[..., Any]) -> bool:
        """Whether ``fn``'s signature can accept an ``agent=`` keyword --
        either a named ``agent`` parameter, or a catch-all ``**kwargs``.

        Unintrospectable callables (some builtins/C-implemented callables
        raise ValueError/TypeError from inspect.signature) are assumed to
        accept it, matching this method's old default (call with agent=
        first) for anything it can't actually inspect.
        """
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return True
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == "agent":
                return True
        return False

    def _call_with_agent_fallback(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call ``fn(*args, **kwargs)``, dropping ``agent=`` first if the
        handler's signature can't accept it (older/narrower handlers).

        Decided via signature inspection BEFORE calling, not by calling
        then retrying on TypeError -- retry-on-TypeError previously meant
        a handler whose OWN body raised an unrelated TypeError (a bug
        inside the handler, nothing to do with its signature) got silently
        called a second time, since that TypeError looked identical to a
        "doesn't accept agent=" signature mismatch. Any side-effecting
        handler (writing a file, incrementing a counter, sending a
        notification) would double its side effect for that single event.
        """
        if "agent" in kwargs and not self._accepts_agent_kwarg(fn):
            kwargs = {k: v for k, v in kwargs.items() if k != "agent"}
        return fn(*args, **kwargs)

    def _fire(self, method: str, *args: Any, **kwargs: Any) -> None:
        for h in self._handlers:
            fn = getattr(h, method, None)
            if not callable(fn):
                continue
            try:
                if inspect.iscoroutinefunction(fn):
                    # No running event loop here (this is the sync-loop
                    # path) -- asyncio.run() gives the coroutine its own
                    # loop to run to completion in.
                    asyncio.run(self._call_with_agent_fallback(fn, *args, **kwargs))
                else:
                    self._call_with_agent_fallback(fn, *args, **kwargs)
            except Exception:
                _logger.warning(
                    "Callback handler %s raised in %s",
                    type(h).__name__,
                    method,
                    exc_info=True,
                )

    async def _afire(self, method: str, *args: Any, **kwargs: Any) -> None:
        for h in self._handlers:
            fn = getattr(h, method, None)
            if not callable(fn):
                continue
            try:
                if inspect.iscoroutinefunction(fn):
                    await self._call_with_agent_fallback(fn, *args, **kwargs)
                else:
                    # Offload the (potentially blocking) sync hook to a
                    # worker thread instead of calling it directly on the
                    # event-loop thread, so e.g. a summarizer middleware's
                    # blocking llm.invoke() inside on_iteration_start
                    # doesn't stall every other concurrent ainvoke() run
                    # sharing this thread for its whole duration.
                    #
                    # Reuse THIS run's captured Context (see
                    # capture_run_context()) so a ContextVar a hook sets
                    # here is visible to a later hook call in the same run
                    # -- run_in_executor itself never propagates contextvar
                    # writes back on its own. Falls back to a bare call
                    # (identical to prior behavior) if nothing captured a
                    # run context (e.g. CallbackManager used directly,
                    # outside Agent.ainvoke()).
                    loop = asyncio.get_running_loop()
                    run_ctx = self._run_context_var.get()
                    if run_ctx is not None:
                        await loop.run_in_executor(
                            self._get_hook_executor(),
                            lambda: run_ctx.run(self._call_with_agent_fallback, fn, *args, **kwargs),
                        )
                    else:
                        await loop.run_in_executor(
                            self._get_hook_executor(),
                            lambda: self._call_with_agent_fallback(fn, *args, **kwargs),
                        )
            except Exception:
                _logger.warning(
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

    def fire_agent_pause(self, iteration: int, reason: Optional[str], agent: Any = None, **kw: Any) -> None:
        self._fire("on_agent_pause", iteration, reason, agent=agent, **kw)

    def fire_agent_resume(self, iteration: int, paused_duration: float, agent: Any = None, **kw: Any) -> None:
        self._fire("on_agent_resume", iteration, paused_duration, agent=agent, **kw)

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
                # See _call_with_agent_fallback's identical comment: decide
                # via signature inspection, not by calling then retrying on
                # TypeError -- a retry-on-TypeError here would silently
                # call fn a second time (with a real side effect, e.g. a
                # screenshot capture or a file write) if its OWN body
                # raised an unrelated TypeError.
                call_kwargs = dict(kw)
                if self._accepts_agent_kwarg(fn):
                    call_kwargs["agent"] = agent
                result = fn(iteration, **call_kwargs)
            except Exception:
                _logger.warning(
                    "Callback handler %s raised in on_before_iteration",
                    type(h).__name__,
                    exc_info=True,
                )
                continue
            if isinstance(result, dict):
                merged.update(result)
        return merged

    def fire_iteration(self, iteration: int, thought: Optional[str], agent: Any = None, **kw: Any) -> bool:
        """
        Calls on_iteration on every handler. Returns True if ANY handler
        signals (via a truthy return) that this iteration's action batch
        should be discarded instead of dispatched; False if none do,
        matching prior behavior (a notification-only hook whose return
        value was always ignored).
        """
        discard = False
        for h in self._handlers:
            fn = getattr(h, "on_iteration", None)
            if not callable(fn):
                continue
            try:
                if inspect.iscoroutinefunction(fn):
                    result = asyncio.run(self._call_with_agent_fallback(fn, iteration, thought, agent=agent, **kw))
                else:
                    result = self._call_with_agent_fallback(fn, iteration, thought, agent=agent, **kw)
            except Exception:
                _logger.warning(
                    "Callback handler %s raised in on_iteration",
                    type(h).__name__,
                    exc_info=True,
                )
                continue
            if result:
                discard = True
        return discard

    def fire_llm_end(self, response: Any, agent: Any = None, **kw: Any) -> None:
        self._fire("on_llm_end", response, agent=agent, **kw)

    def fire_parse_error(self, iteration: int, raw_response: str, agent: Any = None, **kw: Any) -> None:
        self._fire("on_parse_error", iteration, raw_response, agent=agent, **kw)

    # ── async firing (used by the async loops: _arun_loop / _arun_loop_native /
    # _gate_tool_calls_for_approval_async / _execute_tool_async*) ──────────────

    async def afire_agent_start(self, query: str, agent: Any = None, **kw: Any) -> None:
        await self._afire("on_agent_start", query, agent=agent, **kw)

    async def afire_agent_end(self, result: str, agent: Any = None, **kw: Any) -> None:
        await self._afire("on_agent_end", result, agent=agent, **kw)

    async def afire_agent_error(self, error: Exception, agent: Any = None, **kw: Any) -> None:
        await self._afire("on_agent_error", error, agent=agent, **kw)

    async def afire_agent_pause(self, iteration: int, reason: Optional[str], agent: Any = None, **kw: Any) -> None:
        await self._afire("on_agent_pause", iteration, reason, agent=agent, **kw)

    async def afire_agent_resume(self, iteration: int, paused_duration: float, agent: Any = None, **kw: Any) -> None:
        await self._afire("on_agent_resume", iteration, paused_duration, agent=agent, **kw)

    async def afire_tool_start(self, tool_name: str, tool_input: Dict[str, Any], agent: Any = None, **kw: Any) -> None:
        await self._afire("on_tool_start", tool_name, tool_input, agent=agent, **kw)

    async def afire_tool_end(self, tool_name: str, result: str, agent: Any = None, **kw: Any) -> None:
        await self._afire("on_tool_end", tool_name, result, agent=agent, **kw)

    async def afire_tool_error(self, tool_name: str, error: Exception, agent: Any = None, **kw: Any) -> None:
        await self._afire("on_tool_error", tool_name, error, agent=agent, **kw)

    async def afire_iteration_start(self, iteration: int, agent: Any = None, **kw: Any) -> None:
        await self._afire("on_iteration_start", iteration, agent=agent, **kw)

    async def afire_before_iteration(self, iteration: int, agent: Any = None, **kw: Any) -> Dict[str, Any]:
        """Async twin of fire_before_iteration -- same merge semantics, but
        a sync handler's on_before_iteration runs off-thread and an async
        one is awaited directly, instead of always blocking the event loop."""
        merged: Dict[str, Any] = {}
        for h in self._handlers:
            fn = getattr(h, "on_before_iteration", None)
            if not callable(fn):
                continue
            try:
                # See _call_with_agent_fallback's identical comment: decide
                # via signature inspection, not by calling then retrying on
                # TypeError -- a retry-on-TypeError would silently call fn
                # a second time if its OWN body raised an unrelated
                # TypeError, doubling any real side effect.
                call_kwargs = dict(kw)
                if self._accepts_agent_kwarg(fn):
                    call_kwargs["agent"] = agent

                if inspect.iscoroutinefunction(fn):
                    result = await fn(iteration, **call_kwargs)
                else:
                    # See _afire's identical comment: reuse this run's
                    # captured Context (capture_run_context()) so a
                    # ContextVar write here is visible to a later hook call
                    # in the same run; falls back to a bare call when no
                    # run context was captured.
                    loop = asyncio.get_running_loop()
                    run_ctx = self._run_context_var.get()
                    if run_ctx is not None:
                        result = await loop.run_in_executor(
                            self._get_hook_executor(),
                            lambda: run_ctx.run(fn, iteration, **call_kwargs),
                        )
                    else:
                        result = await loop.run_in_executor(
                            self._get_hook_executor(),
                            lambda: fn(iteration, **call_kwargs),
                        )
            except Exception:
                _logger.warning(
                    "Callback handler %s raised in on_before_iteration",
                    type(h).__name__,
                    exc_info=True,
                )
                continue
            if isinstance(result, dict):
                merged.update(result)
        return merged

    async def afire_iteration(self, iteration: int, thought: Optional[str], agent: Any = None, **kw: Any) -> bool:
        """Async twin of fire_iteration -- same discard-signal aggregation,
        but a sync handler's on_iteration runs off-thread (reusing this
        run's captured Context, see _afire's identical comment) and an
        async one is awaited directly, instead of always blocking the
        event loop."""
        discard = False
        for h in self._handlers:
            fn = getattr(h, "on_iteration", None)
            if not callable(fn):
                continue
            try:
                if inspect.iscoroutinefunction(fn):
                    result = await self._call_with_agent_fallback(fn, iteration, thought, agent=agent, **kw)
                else:
                    loop = asyncio.get_running_loop()
                    run_ctx = self._run_context_var.get()
                    if run_ctx is not None:
                        result = await loop.run_in_executor(
                            self._get_hook_executor(),
                            lambda: run_ctx.run(self._call_with_agent_fallback, fn, iteration, thought, agent=agent, **kw),
                        )
                    else:
                        result = await loop.run_in_executor(
                            self._get_hook_executor(),
                            lambda: self._call_with_agent_fallback(fn, iteration, thought, agent=agent, **kw),
                        )
            except Exception:
                _logger.warning(
                    "Callback handler %s raised in on_iteration",
                    type(h).__name__,
                    exc_info=True,
                )
                continue
            if result:
                discard = True
        return discard

    async def afire_llm_end(self, response: Any, agent: Any = None, **kw: Any) -> None:
        await self._afire("on_llm_end", response, agent=agent, **kw)

    async def afire_parse_error(self, iteration: int, raw_response: str, agent: Any = None, **kw: Any) -> None:
        await self._afire("on_parse_error", iteration, raw_response, agent=agent, **kw)


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


# ── built-in scratchpad summarization (Agent(summarize_every=...)) ─────────────
# Inline in the loop, not a CallbackHandler/middleware -- see
# AgentLoopMixin._maybe_summarize/_amaybe_summarize below.

_SUMMARIZE_PROMPT = (
    "You are a context compressor. Summarize the following agent scratchpad into a concise summary "
    "that preserves ALL key findings, tool results, and important observations. Remove redundant "
    "reasoning steps but keep critical data points and intermediate results.\n\n"
    "Original task: {query}\n"
    "Steps completed: {iteration}\n\n"
    "--- SCRATCHPAD TO SUMMARIZE ---\n"
    "{scratchpad}\n"
    "--- END ---\n\n"
    "Provide a concise summary in this format:\n"
    "[Summary of steps 1-{iteration}]\n"
    "Key findings: ...\n"
    "Tool results: ...\n"
    "Current status: ...\n"
)


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

    def _elapsed_excluding_pauses(self, start_time: float) -> float:
        """Wall-clock time since start_time, minus any time this run has
        spent blocked in pause() (see _maybe_pause/_amaybe_pause) -- so
        pausing an agent (e.g. for human input) never counts against
        max_execution_time, matching the same total_paused_time exclusion
        autourgos-hcix's CognitiveInterruptManager already does for its own
        interrupt flow."""
        return time.monotonic() - start_time - getattr(self, "_paused_duration", 0.0)

    def _check_deadline(self, start_time: float, max_exec_time: Optional[float]) -> None:
        """Raise AgentTimeoutError if the run's absolute deadline has already
        passed. Called both at the top of each iteration (existing behavior)
        AND immediately after every blocking wait (an LLM call, a tool-result
        wait, an approval callback) returns -- a blocking call can't be
        preempted mid-flight (sync Python has no way to force-stop a running
        call), but this closes the gap where such a call was previously only
        ever caught one full iteration later, letting the run overrun its
        declared limit by an arbitrary amount.
        """
        if max_exec_time and self._elapsed_excluding_pauses(start_time) > max_exec_time:
            raise AgentTimeoutError(max_exec_time)

    def _inject_agent_deadline(
        self, call_kwargs: Dict[str, Any], start_time: float, max_exec_time: Optional[float]
    ) -> Dict[str, Any]:
        """Add a reserved ``_agent_deadline_seconds`` kwarg (the run's
        remaining time) to a fresh copy of ``call_kwargs``, but ONLY if
        ``self.llm`` explicitly opts in via ``SUPPORTS_AGENT_DEADLINE = True``
        (default False via getattr). This lets an LLM wrapper's own
        retry/fallback loop stop early once the agent is nearly out of time,
        instead of continuing to retry/fall back for its own full budget --
        see OpenAIChatModel.SUPPORTS_AGENT_DEADLINE's docstring for why this
        is opt-in rather than always-on: any other duck-typed BaseLLM that
        forwards unrecognized kwargs straight into request params (as some
        do) would break if it silently received a kwarg it doesn't expect.
        """
        if not max_exec_time or not getattr(self.llm, "SUPPORTS_AGENT_DEADLINE", False):
            return call_kwargs
        remaining = max_exec_time - self._elapsed_excluding_pauses(start_time)
        return {**call_kwargs, "_agent_deadline_seconds": remaining}

    def _maybe_pause(self, iteration: int, cb: "CallbackManager") -> None:
        """Block the current SYNC run at this iteration boundary if
        Agent.pause() has been called and resume() hasn't yet. A no-op
        (returns immediately) when not paused -- the common case, checked
        with a single non-blocking Event.is_set() read. See Agent.pause()/
        resume()/is_paused and README's "Pause & Resume" section."""
        if self._resume_event.is_set():
            return
        reason = self._pause_reason
        cb.fire_agent_pause(iteration, reason, agent=self)
        pause_start = time.monotonic()
        self._resume_event.wait()
        paused_for = time.monotonic() - pause_start
        self._paused_duration += paused_for
        cb.fire_agent_resume(iteration, paused_for, agent=self)

    async def _amaybe_pause(self, iteration: int, cb: "CallbackManager") -> None:
        """Async twin of _maybe_pause. Offloads the blocking
        threading.Event.wait() to a worker thread (via run_in_executor)
        instead of blocking the event loop -- the same Event serves both
        the sync and async loops since pause()/resume() must be safely
        callable from any thread, not necessarily the one running
        invoke()/ainvoke(), and threading.Event's set()/clear()/wait() are
        all thread-safe by construction."""
        if self._resume_event.is_set():
            return
        reason = self._pause_reason
        await cb.afire_agent_pause(iteration, reason, agent=self)
        pause_start = time.monotonic()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._resume_event.wait)
        paused_for = time.monotonic() - pause_start
        self._paused_duration += paused_for
        await cb.afire_agent_resume(iteration, paused_for, agent=self)

    def _summarize_should_trigger(self, iteration: int) -> bool:
        """Cheap (no lock, no LLM call) check: should _do_summarize run for
        this iteration? See Agent(summarize_every=...)."""
        summarize_every = getattr(self, "summarize_every", None)
        if summarize_every is None or not self.scratchpad:
            return False

        # tool_calling_mode="native" never feeds self.scratchpad back to the
        # LLM -- it's kept up to date purely as a human-readable trace (see
        # the native loops' comments); the real conversation state is an
        # internal message list this has no access to. Summarizing
        # scratchpad in that mode would silently burn a real LLM call
        # compressing text the model never sees, with zero effect on the
        # actual context-window budget. Warned once, not skipped silently.
        if getattr(self, "tool_calling_mode", "prompt") == "native":
            if not self._warned_native_summarize:
                self._warned_native_summarize = True
                _logger.warning(
                    "Agent(summarize_every=...): tool_calling_mode is 'native' -- "
                    "agent.scratchpad is not sent to the LLM in native mode, so "
                    "summarizing it has no effect on the actual context-window "
                    "budget. Skipping summarization for this agent."
                )
            return False

        max_chars = getattr(self, "MAX_SCRATCHPAD_CHARS", 15_000)
        current_length = len(self.scratchpad)
        if summarize_every and iteration % summarize_every == 0:
            return True
        if current_length > max_chars:
            # Only re-trigger the char-threshold check if the scratchpad has
            # actually grown since the last summarization -- otherwise a
            # summary that itself stays over max_chars (small threshold, or
            # a verbose summarization LLM) would re-trigger summarization
            # every single iteration even with no new content to compress.
            last_length = self._last_summarized_length
            if last_length is None or current_length > last_length:
                return True
        return False

    def _do_summarize(self, iteration: int) -> None:
        """Actually run summarization for this iteration -- acquires
        _summarizer_lock (non-blocking; a concurrent call for the same
        agent just skips rather than waiting, matching the old middleware's
        behavior), calls the LLM, and writes the result back onto
        self.scratchpad. Call only after _summarize_should_trigger()."""
        if not self._summarizer_lock.acquire(blocking=False):
            _logger.debug("Agent(summarize_every=...): summarization already in progress, skipping.")
            return
        try:
            llm = getattr(self, "summarizer_llm", None) or getattr(self, "llm", None)
            if llm is None:
                _logger.warning("Agent(summarize_every=...): no LLM available. Skipping.")
                return

            original_length = len(self.scratchpad)
            _logger.info(
                f"Triggering auto-summarization at iteration {iteration} "
                f"(scratchpad length: {original_length})."
            )
            prompt = _SUMMARIZE_PROMPT.format(
                query=self.current_query,
                iteration=iteration,
                scratchpad=self.scratchpad,
            )
            try:
                summary = llm.invoke(prompt)
                if hasattr(summary, "content"):
                    summary = summary.content
                summary = str(summary).strip()
                if summary:
                    self.scratchpad = f"[Summarized up to step {iteration}]\n{summary}"
                    self._last_summarized_length = len(self.scratchpad)
                    _logger.info("Agent(summarize_every=...): scratchpad compressed successfully.")
                    logger = getattr(self, "logger", None)
                    if logger:
                        logger.middleware(
                            "Summarizer",
                            f"Compressed scratchpad (iteration {iteration}, was {original_length} chars).",
                        )
                else:
                    _logger.warning(
                        f"Agent(summarize_every=...): summarization LLM returned an empty "
                        f"summary at iteration {iteration}; leaving scratchpad unchanged."
                    )
            except Exception as exc:
                _logger.warning(f"Agent(summarize_every=...): summarization failed: {exc}")
        finally:
            self._summarizer_lock.release()

    def _maybe_summarize(self, iteration: int) -> None:
        """Sync entry point -- see Agent(summarize_every=...)."""
        if self._summarize_should_trigger(iteration):
            self._do_summarize(iteration)

    async def _amaybe_summarize(self, iteration: int) -> None:
        """Async twin of _maybe_summarize. Offloads the blocking LLM call
        to a worker thread (via run_in_executor) instead of blocking the
        event loop, matching how the old middleware's sync hook was
        offloaded when fired from the async loop."""
        if not self._summarize_should_trigger(iteration):
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._do_summarize, iteration)

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
        return _extract_text_fn(raw)

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
                raw_result = func(**tool_input)
            else:
                raw_result = func(tool_input)

            # _execute_tool runs on a worker thread (submitted to a
            # ThreadPoolExecutor by _run_loop/_run_loop_native), so it has no
            # running event loop of its own -- an `async def` tool's call
            # returns an unawaited coroutine here rather than raising, and
            # without this check that coroutine's repr() silently became the
            # tool's "result" (str(coroutine) succeeds), so the tool never
            # actually ran and nothing indicated the failure. asyncio.run()
            # is safe here specifically because this thread has no event
            # loop of its own to conflict with.
            if inspect.isawaitable(raw_result):
                raw_result = asyncio.run(raw_result)

            result = str(raw_result)
        except Exception as exc:
            cb: CallbackManager = getattr(self, "callback_manager", None)
            if cb:
                cb.fire_tool_error(tool_name, exc, agent=self)
            getattr(self, "_history", _NULL_HISTORY).record_tool_error(tool_name, exc)
            result = f"Error executing '{tool_name}': {exc}"

        if len(result) > max_chars:
            result = result[:max_chars] + "... [truncated]"

        return result

    async def _execute_tool_async(
        self, tool_map: Dict[str, Any], tool_name: str, tool_input: Any,
        pool: Optional[ThreadPoolExecutor] = None,
    ) -> str:
        """Async version of _execute_tool — awaits coroutine funcs directly;
        dispatches a plain sync func to `pool` (via run_in_executor) instead
        of calling it inline, so a blocking tool doesn't stall the event
        loop for every other concurrently-running task/iteration. `pool` is
        the run's shared, MAX_TOOL_WORKERS-bounded executor (see _arun_loop /
        _arun_loop_native) -- when omitted (e.g. direct unit-test calls),
        falls back to the loop's default executor, same as calling with no
        pool ever did implicitly."""
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
            if inspect.iscoroutinefunction(func):
                if isinstance(tool_input, dict):
                    raw_result = await func(**tool_input)
                else:
                    raw_result = await func(tool_input)
            else:
                loop = asyncio.get_running_loop()
                if isinstance(tool_input, dict):
                    raw_result = await loop.run_in_executor(pool, lambda: func(**tool_input))
                else:
                    raw_result = await loop.run_in_executor(pool, func, tool_input)

            if inspect.isawaitable(raw_result):
                raw_result = await raw_result

            result = str(raw_result)
        except Exception as exc:
            cb: CallbackManager = getattr(self, "callback_manager", None)
            if cb:
                await cb.afire_tool_error(tool_name, exc, agent=self)
            getattr(self, "_history", _NULL_HISTORY).record_tool_error(tool_name, exc)
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

        def _on_retry(exc: BaseException, attempt: int, delay: float) -> None:
            if logger:
                logger.info(
                    f"LLM call failed ({exc}); retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{retries})."
                )

        return retry_with_backoff(
            fn, max_attempts=retries + 1, backoff_base=backoff, max_backoff=max_backoff,
            should_retry=should_retry, on_retry=_on_retry,
        )

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

        def _on_retry(exc: BaseException, attempt: int, delay: float) -> None:
            if logger:
                logger.info(
                    f"LLM call failed ({exc}); retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{retries})."
                )

        return await aretry_with_backoff(
            coro_fn, max_attempts=retries + 1, backoff_base=backoff, max_backoff=max_backoff,
            should_retry=should_retry, on_retry=_on_retry,
        )

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
            getattr(self, "_history", _NULL_HISTORY).record_tool_error(tool_name, TimeoutError(result))
            return result

    async def _execute_tool_async_with_timeout(
        self, tool_map: Dict[str, Any], tool_name: str, tool_input: Any, timeout: Optional[float],
        pool: Optional[ThreadPoolExecutor] = None,
    ) -> str:
        """Async twin of _collect_future_result -- wraps _execute_tool_async in
        asyncio.wait_for so a hung async tool doesn't block the loop forever.
        For a genuinely async tool func, wait_for can actually cancel it at
        its next await point; for a sync tool func running in `pool`
        (see _execute_tool_async), the wait_for timeout still fires on
        schedule since waiting on the executor future doesn't block the
        loop, but the underlying worker thread itself is not interrupted --
        same abandon-in-place handling as the sync path's _collect_future_result.
        """
        try:
            return await asyncio.wait_for(
                self._execute_tool_async(tool_map, tool_name, tool_input, pool=pool), timeout=timeout
            )
        except asyncio.TimeoutError:
            result = f"Error: tool '{tool_name}' timed out after {timeout}s."
            cb: CallbackManager = getattr(self, "callback_manager", None)
            if cb:
                await cb.afire_tool_error(tool_name, TimeoutError(result), agent=self)
            getattr(self, "_history", _NULL_HISTORY).record_tool_error(tool_name, TimeoutError(result))
            return result

    def _trim_text_to_budget(self, text: str, marker: str = "[...earlier steps trimmed...]\n") -> str:
        """Trim `text` to fit both the character cap (MAX_SCRATCHPAD_CHARS,
        always active) and, if set, a token budget (max_scratchpad_tokens) --
        char count alone is a poor proxy for what actually overflows an LLM's
        context window, since tokens-per-char varies a lot by language and
        content (dense non-English text or code can run well under 4
        chars/token, silently blowing a char-only budget's whole point).

        Shared by _trim_scratchpad (the prompt-mode scratchpad) and the
        memory-context trim used by both prompt and native modes, so a
        memory object's format_for_llm() output is bounded by the same
        limits instead of being injected into the final prompt/messages
        completely unbounded (previously the case -- only the scratchpad
        itself was ever capped here).
        """
        max_chars: int = getattr(self, "MAX_SCRATCHPAD_CHARS", 15000)
        if len(text) > max_chars:
            text = marker + text[-max_chars:]

        max_tokens: Optional[int] = getattr(self, "max_scratchpad_tokens", None)
        if max_tokens is not None:
            counter: Callable[[str], int] = getattr(self, "token_counter", None) or _default_token_counter
            if counter(text) > max_tokens:
                text = _trim_to_token_budget(text, max_tokens, counter, prefix=marker)

        return text

    def _trim_scratchpad(self, scratchpad: str) -> str:
        """Trim the scratchpad to the shared context budget. See
        _trim_text_to_budget for the actual char/token trimming logic."""
        return self._trim_text_to_budget(scratchpad)

    def _trim_memory_context(self, memory_context: str) -> str:
        """Trim memory's format_for_llm() output to the same context budget
        _trim_scratchpad enforces.

        In prompt mode this is capped independently of the scratchpad's own
        budget usage, not combined into one shared total -- doing so would
        require restructuring the prompt template's separate
        {memory_context}/{previous_context} placeholders into one budgeted
        blob, a larger change than closing the "memory_context is
        completely unbounded" gap warrants. Worst case here is roughly 2x
        MAX_SCRATCHPAD_CHARS (scratchpad + memory_context each capped
        separately) instead of unbounded.

        Native mode does better: _trim_native_messages additionally
        coordinates this with the turns budget, since there the whole
        call_messages list is assembled in one place already, so the real
        combined total (system_messages + trimmed turns) stays within one
        budget.
        """
        return self._trim_text_to_budget(memory_context, marker="[...older memory trimmed...]\n")

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
        memory_context = self._trim_memory_context(_get_memory_context(getattr(self, "memory", None), query))

        # One bounded pool for the whole run, not a fresh one per iteration --
        # a fresh pool each iteration meant MAX_TOOL_WORKERS only ever capped
        # that iteration's tool calls, not total in-flight tool threads across
        # a multi-iteration run with repeated timeouts (each abandoned thread
        # from a prior iteration's timed-out tool, see _collect_future_result,
        # kept accumulating alongside brand-new full-size pools). shutdown
        # happens once, in the finally below, not after every iteration.
        pool = ThreadPoolExecutor(max_workers=max(1, getattr(self, "MAX_TOOL_WORKERS", 8)))
        try:
          for iteration in range(1, max_iterations + 1):
            cb.fire_iteration_start(iteration, agent=self)
            self._history.begin_iteration(iteration)
            self._maybe_pause(iteration, cb)
            self._maybe_summarize(iteration)
            preiteration_kwargs = self._preiteration.before_iteration(iteration, agent=self)
            iteration_extra_kwargs = cb.fire_before_iteration(iteration, agent=self)
            if preiteration_kwargs:
                iteration_extra_kwargs = {**preiteration_kwargs, **iteration_extra_kwargs}

            # time guard
            if max_exec_time and self._elapsed_excluding_pauses(start_time) > max_exec_time:
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
            call_kwargs = self._inject_agent_deadline(
                {**extra_kwargs, **iteration_extra_kwargs}, start_time, max_exec_time
            )
            try:
                raw = self._call_llm_with_retry(lambda: self.llm.invoke(messages, **call_kwargs))  # type: ignore[attr-defined]
                response_text = self._extract_text(raw)
            except Exception as exc:
                raise AgentLLMError(exc) from exc

            # Recheck immediately after the call returns -- see
            # _check_deadline's docstring: a hung/slow call already
            # completed can still have blown the deadline while it ran.
            self._check_deadline(start_time, max_exec_time)

            cb.fire_llm_end(response_text, agent=self, raw=raw, **self._extract_llm_metadata(raw))
            self._history.record_thought(response_text)

            if logger and getattr(logger, "full_output", False):
                logger.llm_response(response_text, iteration)

            # parse
            try:
                thought, actions, final_answer = self._parser(response_text)  # type: ignore[attr-defined]
            except Exception:
                thought, actions, final_answer = None, [], None

            # Fired unconditionally (not gated on `if thought:`) -- a
            # handler (e.g. HcixInterruptMiddleware) needs the chance to
            # signal "discard this turn's action batch" on every turn that
            # might dispatch actions, not only turns that also produced a
            # thought.
            discard_batch = cb.fire_iteration(iteration, thought, agent=self)
            if thought and logger:
                logger.thought(thought, iteration)

            # final answer
            if final_answer:
                memory = getattr(self, "memory", None)
                if memory:
                    _record_agent_message(memory, final_answer)
                cb.fire_agent_end(final_answer, agent=self)
                self._history.finish(final_answer)
                self._preiteration.cleanup()
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

            # FRAMEWORK_REVIEW.md Finding #5: a handler that just injected a
            # newer human instruction (e.g. HcixInterruptMiddleware.on_iteration)
            # can signal via its return value that the actions this LLM turn
            # already produced were reasoned out BEFORE that instruction
            # arrived, and are therefore stale -- discard them unexecuted
            # rather than dispatching a plan the human has already
            # superseded, and let the next iteration's LLM call (which will
            # see the injected override) produce a fresh one. This is
            # deliberately NOT treated as a parse error (consecutive_parse_errors
            # is already reset to 0 above): a run that legitimately keeps
            # getting interrupted must never trip AgentParseError/
            # AgentMaxIterationsError just for that.
            if discard_batch:
                self.scratchpad += (
                    f"\nStep {iteration}:\n"
                    f"Thought: {thought or 'None'}\n"
                    f"Observation: A newer human instruction was received; "
                    f"the previous action plan was discarded before execution.\n"
                )
                self.scratchpad = self._trim_scratchpad(self.scratchpad)
                continue

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
                self._history.record_tool_start(tool_name, tool_input)

                # approval gate
                if approval_callback:
                    is_approved = _call_sync_approval(approval_callback, tool_name, tool_input)
                    self._check_deadline(start_time, max_exec_time)
                else:
                    is_approved = True
                if approval_callback and not is_approved:
                    denial_result = "Tool call was denied by the approval callback."
                    cb.fire_tool_end(tool_name, denial_result, agent=self)
                    self._history.record_tool_result(tool_name, denial_result)
                    step_lines.append(f"Action: {tool_name}({tool_input})")
                    step_lines.append(f"Observation: {denial_result}")
                    continue

                approved.append((tool_name, tool_input))

            if approved:
                # Reuses the one pool created for the whole run (see above) --
                # sized to MAX_TOOL_WORKERS regardless of how many tools this
                # particular iteration approved, so the cap holds across
                # iterations, not just within one.
                futures = [
                    (tool_name, tool_input, pool.submit(self._execute_tool, tool_map, tool_name, tool_input))
                    for tool_name, tool_input in approved
                ]
                for tool_name, tool_input, future in futures:
                    result = self._collect_future_result(tool_name, future, tool_timeout)
                    self._check_deadline(start_time, max_exec_time)
                    cb.fire_tool_end(tool_name, result, agent=self)
                    self._history.record_tool_result(tool_name, result)
                    if logger:
                        logger.tool_result(tool_name, result, iteration)
                    step_lines.append(f"Action: {tool_name}({tool_input})")
                    step_lines.append(f"Observation: {result}")

            self.scratchpad += "\n".join(step_lines)
            self.scratchpad = self._trim_scratchpad(self.scratchpad)

          raise AgentMaxIterationsError(max_iterations)
        finally:
            # Not `with pool:` -- that calls shutdown(wait=True), which would
            # block here until every submitted thread finishes, including one
            # _collect_future_result already gave up on via tool_timeout.
            # wait=False abandons any still-running (timed-out) thread to
            # finish on its own instead of blocking run teardown on it.
            pool.shutdown(wait=False)

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
        memory_context = self._trim_memory_context(_get_memory_context(getattr(self, "memory", None), query))

        # One semaphore for the whole run so MAX_TOOL_WORKERS actually caps
        # concurrent async tool calls -- previously unenforced here (plain
        # asyncio.gather with no cap at all), unlike the sync loop above.
        tool_semaphore = asyncio.Semaphore(max(1, getattr(self, "MAX_TOOL_WORKERS", 8)))

        # Bounded executor a sync tool func is dispatched to (see
        # _execute_tool_async) instead of running inline on the event loop
        # thread -- one per run, reused across iterations, shut down in the
        # finally below regardless of how the run ends.
        tool_pool = ThreadPoolExecutor(max_workers=max(1, getattr(self, "MAX_TOOL_WORKERS", 8)))
        try:
          for iteration in range(1, max_iterations + 1):
            await cb.afire_iteration_start(iteration, agent=self)
            self._history.begin_iteration(iteration)
            await self._amaybe_pause(iteration, cb)
            await self._amaybe_summarize(iteration)
            preiteration_kwargs = await self._preiteration.abefore_iteration(iteration, agent=self)
            iteration_extra_kwargs = await cb.afire_before_iteration(iteration, agent=self)
            if preiteration_kwargs:
                iteration_extra_kwargs = {**preiteration_kwargs, **iteration_extra_kwargs}

            if max_exec_time and self._elapsed_excluding_pauses(start_time) > max_exec_time:
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

            call_kwargs = self._inject_agent_deadline(
                {**extra_kwargs, **iteration_extra_kwargs}, start_time, max_exec_time
            )
            try:
                acall = self._acall_llm_with_retry(lambda: self.llm.ainvoke(messages, **call_kwargs))  # type: ignore[attr-defined]
                if max_exec_time:
                    remaining = max_exec_time - self._elapsed_excluding_pauses(start_time)
                    if remaining <= 0:
                        raise AgentTimeoutError(max_exec_time)
                    raw = await asyncio.wait_for(acall, timeout=remaining)
                else:
                    raw = await acall
                response_text = self._extract_text(raw)
            except AgentTimeoutError:
                raise
            except asyncio.TimeoutError:
                raise AgentTimeoutError(max_exec_time)
            except Exception as exc:
                raise AgentLLMError(exc) from exc

            # Recheck immediately after the call returns -- see _run_loop's
            # identical comment: a call that finished just under wait_for's
            # timeout could still have blown the deadline while it ran.
            self._check_deadline(start_time, max_exec_time)

            await cb.afire_llm_end(response_text, agent=self, raw=raw, **self._extract_llm_metadata(raw))
            self._history.record_thought(response_text)

            if logger and getattr(logger, "full_output", False):
                logger.llm_response(response_text, iteration)

            try:
                thought, actions, final_answer = self._parser(response_text)  # type: ignore[attr-defined]
            except Exception:
                thought, actions, final_answer = None, [], None

            # See _run_loop's identical comment: fired unconditionally so
            # the discard signal can apply even on a turn with no thought.
            discard_batch = await cb.afire_iteration(iteration, thought, agent=self)
            if thought and logger:
                logger.thought(thought, iteration)

            if final_answer:
                memory = getattr(self, "memory", None)
                if memory:
                    _record_agent_message(memory, final_answer)
                await cb.afire_agent_end(final_answer, agent=self)
                self._history.finish(final_answer)
                self._preiteration.cleanup()
                if logger:
                    logger.final_answer(final_answer)
                return final_answer

            if not actions:
                # see _run_loop's identical comment: must only reset once a
                # turn actually produces actions/final_answer, not on every
                # non-throwing _parser() call, or this can never trip.
                consecutive_parse_errors += 1
                await cb.afire_parse_error(iteration, response_text, agent=self)
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

            # See _run_loop's identical comment (Finding #5): discard a
            # stale action batch instead of dispatching it.
            if discard_batch:
                self.scratchpad += (
                    f"\nStep {iteration}:\n"
                    f"Thought: {thought or 'None'}\n"
                    f"Observation: A newer human instruction was received; "
                    f"the previous action plan was discarded before execution.\n"
                )
                self.scratchpad = self._trim_scratchpad(self.scratchpad)
                continue

            step_lines: List[str] = [f"\nStep {iteration}:"]
            if thought:
                step_lines.append(f"Thought: {thought}")

            approved: List[Tuple[str, Any]] = []
            for action_dict in actions:
                tool_name  = action_dict.get("action", "")
                tool_input = action_dict.get("action_input", {})

                if logger:
                    logger.tool_call(tool_name, tool_input, iteration)
                await cb.afire_tool_start(tool_name, tool_input, agent=self)
                self._history.record_tool_start(tool_name, tool_input)

                if approval_callback:
                    is_approved = await _maybe_await(approval_callback(tool_name, tool_input))
                    self._check_deadline(start_time, max_exec_time)
                    if not is_approved:
                        denial_result = "Tool call was denied by the approval callback."
                        await cb.afire_tool_end(tool_name, denial_result, agent=self)
                        self._history.record_tool_result(tool_name, denial_result)
                        step_lines.append(f"Action: {tool_name}({tool_input})")
                        step_lines.append(f"Observation: {denial_result}")
                        continue

                approved.append((tool_name, tool_input))

            if approved:
                async def _run_bounded(tool_name: str, tool_input: Any) -> str:
                    async with tool_semaphore:
                        return await self._execute_tool_async_with_timeout(
                            tool_map, tool_name, tool_input, tool_timeout, pool=tool_pool
                        )

                results = await asyncio.gather(*[
                    _run_bounded(tool_name, tool_input)
                    for tool_name, tool_input in approved
                ])
                self._check_deadline(start_time, max_exec_time)
                for (tool_name, tool_input), result in zip(approved, results):
                    await cb.afire_tool_end(tool_name, result, agent=self)
                    self._history.record_tool_result(tool_name, result)
                    if logger:
                        logger.tool_result(tool_name, result, iteration)
                    step_lines.append(f"Action: {tool_name}({tool_input})")
                    step_lines.append(f"Observation: {result}")

            self.scratchpad += "\n".join(step_lines)
            self.scratchpad = self._trim_scratchpad(self.scratchpad)

          raise AgentMaxIterationsError(max_iterations)
        finally:
            tool_pool.shutdown(wait=False)

    # ── native tool-calling loop (tool_calling_mode="native") ────────────────
    # Uses the LLM's own invoke_with_tools()/ainvoke_with_tools() -- structured
    # tool_calls straight from the API -- instead of hand-rolled JSON-in-text
    # parsing. Conversation state is a real multi-turn message list, not a
    # single rendered scratchpad string; self.scratchpad is still kept up to
    # date (human-readable trace only, not fed back to the LLM) so middleware
    # relying on the scratchpad contract still sees something sensible.

    def _build_native_messages(self, query: str) -> List[Dict[str, Any]]:
        """The conversation-turn part of the native-mode message list --
        deliberately excludes the system prompt, which is no longer baked
        in here. See _native_system_messages()."""
        return [{"role": "user", "content": query}]

    def _native_system_messages(self, memory_context: str = "") -> List[Dict[str, Any]]:
        """Build the system-role prefix fresh from the *current*
        self.system_prompt on every call, instead of _build_native_messages'
        old approach of baking it into the message list once before the
        loop started. Native mode's `messages` list is otherwise never
        touched by anything that only knows how to write to
        agent.system_prompt (autourgos-hcix's human-override injection,
        autourgos-toolbox's "tools were just unlocked" notice) -- those
        middleware packages mutate agent.system_prompt expecting the agent
        to pick it up on the very next LLM call, which is true in prompt
        mode (the prompt is re-rendered from self.system_prompt every
        iteration) but was never true in native mode until this. Called
        fresh every iteration and prepended to the conversation turns
        rather than merged into them, so mid-run edits reach the model
        starting the next iteration, matching prompt mode's behavior.
        """
        system_prompt: str = getattr(self, "system_prompt", "") or ""
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if memory_context:
            messages.append({"role": "system", "content": memory_context})
        return messages

    def _group_native_turns(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group messages[1:] (everything after the original user query)
        into turns: an assistant/user message plus any tool-result messages
        immediately following it. Trimming must drop whole turns, never a
        lone message out of the middle of one -- dropping only the
        assistant message that made a tool call while leaving its "tool"
        role results behind would orphan a tool_call_id the API doesn't
        recognize and most providers reject the request outright.
        """
        turns: List[List[Dict[str, Any]]] = []
        i = 1
        while i < len(messages):
            turn = [messages[i]]
            i += 1
            while i < len(messages) and messages[i].get("role") == "tool":
                turn.append(messages[i])
                i += 1
            turns.append(turn)
        return turns

    def _trim_native_messages(
        self, messages: List[Dict[str, Any]], system_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Keep the native-mode conversation list within the same budget
        _trim_scratchpad enforces for prompt mode (MAX_SCRATCHPAD_CHARS,
        plus max_scratchpad_tokens if set) -- native mode's `messages` list
        otherwise grows unboundedly across iterations (nothing was ever
        capping it) until it blows the model's context window. Drops
        whole turns (see _group_native_turns) from the oldest end,
        always keeping the original user query (messages[0]) and at least
        one turn so the conversation never goes fully empty.

        `system_messages` -- the system_prompt + (already independently
        trimmed) memory_context that the caller will prepend to form the
        actual `call_messages` sent to the LLM -- is counted against this
        same budget when given, so the real final total (system_messages +
        trimmed turns) is coordinated against one budget instead of the
        turns being trimmed against the full budget while system_messages
        gets added on top of it uncounted. Falls back to using the whole
        budget for turns alone if omitted, matching prior behavior.
        """
        if len(messages) <= 1:
            return messages

        max_chars: int = getattr(self, "MAX_SCRATCHPAD_CHARS", 15000)
        max_tokens: Optional[int] = getattr(self, "max_scratchpad_tokens", None)
        counter: Callable[[str], int] = getattr(self, "token_counter", None) or _default_token_counter

        reserved_chars = len(json.dumps(system_messages, default=str)) if system_messages else 0
        turn_max_chars = max(max_chars - reserved_chars, 0)

        head = messages[:1]
        turns = self._group_native_turns(messages)

        def _flatten() -> List[Dict[str, Any]]:
            return head + [m for turn in turns for m in turn]

        while len(turns) > 1 and len(json.dumps(_flatten(), default=str)) > turn_max_chars:
            turns.pop(0)

        if max_tokens is not None:
            reserved_tokens = counter(json.dumps(system_messages, default=str)) if system_messages else 0
            turn_max_tokens = max(max_tokens - reserved_tokens, 0)
            while len(turns) > 1 and counter(json.dumps(_flatten(), default=str)) > turn_max_tokens:
                turns.pop(0)

        return _flatten()

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
            self._history.record_tool_start(tc.name, tc.arguments)
            if approval_callback and not _call_sync_approval(approval_callback, tc.name, tc.arguments):
                result = "Tool call was denied by the approval callback."
                cb.fire_tool_end(tc.name, result, agent=self)
                self._history.record_tool_result(tc.name, result)
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
            await cb.afire_tool_start(tc.name, tc.arguments, agent=self)
            self._history.record_tool_start(tc.name, tc.arguments)
            if approval_callback and not await _maybe_await(approval_callback(tc.name, tc.arguments)):
                result = "Tool call was denied by the approval callback."
                await cb.afire_tool_end(tc.name, result, agent=self)
                self._history.record_tool_result(tc.name, result)
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
        memory_context = self._trim_memory_context(_get_memory_context(getattr(self, "memory", None), query))
        messages = self._build_native_messages(query)

        # See _run_loop's identical comment: one bounded pool for the whole
        # run, reused across iterations, rather than a fresh one each time.
        pool = ThreadPoolExecutor(max_workers=max(1, getattr(self, "MAX_TOOL_WORKERS", 8)))
        try:
          for iteration in range(1, max_iterations + 1):
            cb.fire_iteration_start(iteration, agent=self)
            self._history.begin_iteration(iteration)
            self._maybe_pause(iteration, cb)
            self._maybe_summarize(iteration)
            preiteration_kwargs = self._preiteration.before_iteration(iteration, agent=self)
            iteration_extra_kwargs = cb.fire_before_iteration(iteration, agent=self)
            if preiteration_kwargs:
                iteration_extra_kwargs = {**preiteration_kwargs, **iteration_extra_kwargs}

            if max_exec_time and self._elapsed_excluding_pauses(start_time) > max_exec_time:
                raise AgentTimeoutError(max_exec_time)

            # Rebuilt every iteration -- see _run_loop's identical comment.
            # self.tools is already passed live to invoke_with_tools() below,
            # but tool_map (used to actually execute an approved call further
            # down) was previously frozen once before the loop, so a tool
            # exposed mid-run could be advertised to the LLM yet still fail
            # with "not found" the moment it tried to call it.
            tool_map: Dict[str, Any] = {_tool_name(t): t for t in getattr(self, "tools", [])}

            system_messages = self._native_system_messages(memory_context)
            messages = self._trim_native_messages(messages, system_messages=system_messages)
            call_messages = system_messages + messages

            call_kwargs = self._inject_agent_deadline(
                {**extra_kwargs, **iteration_extra_kwargs}, start_time, max_exec_time
            )
            try:
                response = self._call_llm_with_retry(
                    lambda: self.llm.invoke_with_tools(call_messages, self.tools, **call_kwargs)  # type: ignore[attr-defined]
                )
            except NotImplementedError as exc:
                raise self._wrap_unsupported_native_error(exc) from exc
            except Exception as exc:
                raise AgentLLMError(exc) from exc

            # Recheck immediately after the call returns -- see _run_loop's
            # identical comment.
            self._check_deadline(start_time, max_exec_time)

            cb.fire_llm_end(
                response.text if response.is_final_answer else None,
                agent=self,
                raw=response.raw,
                **self._extract_llm_metadata(response.raw),
            )
            self._history.record_thought(response.text if response.is_final_answer else None)

            if response.is_final_answer:
                final_answer = response.text
                memory = getattr(self, "memory", None)
                if memory:
                    _record_agent_message(memory, final_answer)
                cb.fire_agent_end(final_answer, agent=self)
                self._history.finish(final_answer)
                self._preiteration.cleanup()
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
            self._check_deadline(start_time, max_exec_time)

            if approved:
                # Reuses the run's one pool -- see _run_loop's identical note.
                futures = [(tc, pool.submit(self._execute_tool, tool_map, tc.name, tc.arguments)) for tc in approved]
                for tc, future in futures:
                    result = self._collect_future_result(tc.name, future, tool_timeout)
                    self._check_deadline(start_time, max_exec_time)
                    cb.fire_tool_end(tc.name, result, agent=self)
                    self._history.record_tool_result(tc.name, result)
                    if logger:
                        logger.tool_result(tc.name, result, iteration)
                    calls_and_results.append((tc, result))

            results_by_call_id = {tc.call_id: result for tc, result in calls_and_results}
            for tc in response.tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": results_by_call_id.get(tc.call_id, "")})

            self._record_native_step(iteration, calls_and_results)

          raise AgentMaxIterationsError(max_iterations)
        finally:
            pool.shutdown(wait=False)

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
        memory_context = self._trim_memory_context(_get_memory_context(getattr(self, "memory", None), query))
        messages = self._build_native_messages(query)

        # See _arun_loop's identical comment.
        tool_semaphore = asyncio.Semaphore(max(1, getattr(self, "MAX_TOOL_WORKERS", 8)))

        # See _arun_loop's identical comment: bounded executor sync tool
        # funcs are dispatched to instead of running inline on the loop.
        tool_pool = ThreadPoolExecutor(max_workers=max(1, getattr(self, "MAX_TOOL_WORKERS", 8)))
        try:
          for iteration in range(1, max_iterations + 1):
            await cb.afire_iteration_start(iteration, agent=self)
            self._history.begin_iteration(iteration)
            await self._amaybe_pause(iteration, cb)
            await self._amaybe_summarize(iteration)
            preiteration_kwargs = await self._preiteration.abefore_iteration(iteration, agent=self)
            iteration_extra_kwargs = await cb.afire_before_iteration(iteration, agent=self)
            if preiteration_kwargs:
                iteration_extra_kwargs = {**preiteration_kwargs, **iteration_extra_kwargs}

            if max_exec_time and self._elapsed_excluding_pauses(start_time) > max_exec_time:
                raise AgentTimeoutError(max_exec_time)

            # Rebuilt every iteration -- see _run_loop_native's identical comment.
            tool_map: Dict[str, Any] = {_tool_name(t): t for t in getattr(self, "tools", [])}

            system_messages = self._native_system_messages(memory_context)
            messages = self._trim_native_messages(messages, system_messages=system_messages)
            call_messages = system_messages + messages

            call_kwargs = self._inject_agent_deadline(
                {**extra_kwargs, **iteration_extra_kwargs}, start_time, max_exec_time
            )
            try:
                acall = self._acall_llm_with_retry(
                    lambda: self.llm.ainvoke_with_tools(call_messages, self.tools, **call_kwargs)  # type: ignore[attr-defined]
                )
                if max_exec_time:
                    remaining = max_exec_time - self._elapsed_excluding_pauses(start_time)
                    if remaining <= 0:
                        raise AgentTimeoutError(max_exec_time)
                    response = await asyncio.wait_for(acall, timeout=remaining)
                else:
                    response = await acall
            except NotImplementedError as exc:
                raise self._wrap_unsupported_native_error(exc) from exc
            except AgentTimeoutError:
                raise
            except asyncio.TimeoutError:
                raise AgentTimeoutError(max_exec_time)
            except Exception as exc:
                raise AgentLLMError(exc) from exc

            # Recheck immediately after the call returns -- see _run_loop's
            # identical comment.
            self._check_deadline(start_time, max_exec_time)

            await cb.afire_llm_end(
                response.text if response.is_final_answer else None,
                agent=self,
                raw=response.raw,
                **self._extract_llm_metadata(response.raw),
            )
            self._history.record_thought(response.text if response.is_final_answer else None)

            if response.is_final_answer:
                final_answer = response.text
                memory = getattr(self, "memory", None)
                if memory:
                    _record_agent_message(memory, final_answer)
                await cb.afire_agent_end(final_answer, agent=self)
                self._history.finish(final_answer)
                self._preiteration.cleanup()
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
            self._check_deadline(start_time, max_exec_time)

            if approved:
                async def _run_bounded(tc: Any) -> str:
                    async with tool_semaphore:
                        return await self._execute_tool_async_with_timeout(
                            tool_map, tc.name, tc.arguments, tool_timeout, pool=tool_pool
                        )

                results = await asyncio.gather(*[_run_bounded(tc) for tc in approved])
                self._check_deadline(start_time, max_exec_time)
                for tc, result in zip(approved, results):
                    await cb.afire_tool_end(tc.name, result, agent=self)
                    self._history.record_tool_result(tc.name, result)
                    if logger:
                        logger.tool_result(tc.name, result, iteration)
                    calls_and_results.append((tc, result))

            results_by_call_id = {tc.call_id: result for tc, result in calls_and_results}
            for tc in response.tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": results_by_call_id.get(tc.call_id, "")})

            self._record_native_step(iteration, calls_and_results)

          raise AgentMaxIterationsError(max_iterations)
        finally:
            tool_pool.shutdown(wait=False)


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

        # Guards against two concurrent invoke()/ainvoke() calls on the same
        # instance stomping on the shared mid-run state above. A plain Lock
        # is safe from both the sync and async entry points: the critical
        # section is just a flag check-and-set, never held across an await.
        self._run_active: bool = False
        self._run_lock = threading.Lock()

        # Pause/resume support (in-process blocking) -- see README's "Pause
        # & Resume" section. A plain threading.Event (not asyncio.Event):
        # pause()/resume() must be safely callable from ANY thread, not
        # necessarily the one running invoke()/ainvoke(), and Event's
        # set()/clear()/wait() are thread-safe by construction -- the async
        # loop offloads .wait() to a worker thread (AgentLoopMixin.
        # _amaybe_pause) instead of blocking the event loop, so this one
        # Event serves both loop flavors. Starts "set" (not paused).
        self._resume_event: threading.Event = threading.Event()
        self._resume_event.set()
        self._paused_duration: float = 0.0
        self._pause_reason: Optional[str] = None

    def pause(self, reason: Optional[str] = None) -> "BaseAgent":
        """Request a pause: the running loop (if any) blocks at its next
        iteration boundary -- before the next LLM call, never mid-tool-call
        or mid-LLM-call -- until resume() is called. Thread-safe: call it
        from any thread, including one other than the thread invoke()/
        ainvoke() is running on. Calling pause() before invoke()/ainvoke()
        starts means that run begins already paused (blocks before its
        first iteration) -- not a bug, a valid way to start an agent
        pre-paused. Time spent paused does not count against
        max_execution_time. See README's "Pause & Resume" section.
        """
        self._pause_reason = reason
        self._resume_event.clear()
        return self

    def resume(self) -> "BaseAgent":
        """Release a pending/active pause. No-op if not currently paused."""
        self._pause_reason = None
        self._resume_event.set()
        return self

    @property
    def is_paused(self) -> bool:
        """True from the moment pause() is called until resume() is called
        -- regardless of whether the loop has actually reached a blocking
        point yet (it may still be mid-LLM-call, which pause() never
        interrupts)."""
        return not self._resume_event.is_set()

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
