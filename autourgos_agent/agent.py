"""
Agent Agent (Reasoning + Acting)

An advanced AI agent that combines reasoning and tool execution in an
iterative loop.  The agent thinks through problems step-by-step while
taking actions using the tools you provide.

The agent loop
--------------
  1. Render the prompt with the current scratchpad and the user query.
  2. Call the LLM — get back a JSON object with thought / actions / final_answer.
  3. If final_answer is set  → return it.
  4. If actions are present  → execute each tool, append results to the scratchpad.
  5. Repeat until final_answer is found or max_iterations is reached.

Works with ANY OpenAI-compatible LLM
-------------------------------------
  from autourgos_openaichat import OpenAIChatModel   # Chat Completions API
  from autourgos_responses   import OpenAIResponse   # Responses API
  # or any object with .invoke() / .ainvoke()

  agent = Agent(llm=OpenAIChatModel(model="gpt-4o"))
"""

from __future__ import annotations

import inspect
import threading
from typing import Any, Callable, Dict, List, Optional, Union

from autourgos_core import Toolbox

from ._preiteration import _NULL_PREITERATION, _PreIterationRuntime
from ._toolbox import ToolboxMiddleware
from .base import (
    AgentAlreadyRunningError,
    AgentLoopMixin,
    BaseAgent,
    CallbackHandler,
    CallbackManager,
    MemoryProtocol,
)
from .history import _NULL_HISTORY, _HistoryRecorder
from .logging import AgentLogger
from .prompt import LOGIC_PROMPT, PREFIX_PROMPT, SUFFIX_PROMPT
from .runtime import parse_json_object


class _FunctionStartHandler(CallbackHandler):
    """Wraps a plain function passed as `Agent(on_agent_start=...)` into a
    CallbackHandler, so it goes through the same CallbackManager machinery
    (sync/async bridging, agent-kwarg detection) as any other middleware.

    CallbackManager decides sync-vs-async dispatch by checking
    inspect.iscoroutinefunction() on the *handler method itself*
    (base.py's _fire/_afire: `getattr(h, method)` then that check) -- it
    never awaits a coroutine returned from an ordinary ``def``. So if `fn`
    is an `async def`, on_agent_start must ALSO be declared `async def`
    (not just call an async fn from a sync one), or the coroutine `fn(...)`
    returns would silently never run -- the same footgun the module
    docstring for _call_sync_approval warns about for approval_callback.
    Building the method per-instance (rather than one fixed method that
    awaits conditionally) is what lets iscoroutinefunction see the right
    answer for whichever kind of `fn` was passed.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn
        wants_agent = CallbackManager._accepts_agent_kwarg(fn)

        if inspect.iscoroutinefunction(fn):
            async def on_agent_start(query: str, agent: Any = None, **kwargs: Any) -> None:
                if wants_agent:
                    await fn(query, agent=agent)
                else:
                    await fn(query)
        else:
            def on_agent_start(query: str, agent: Any = None, **kwargs: Any) -> None:
                if wants_agent:
                    fn(query, agent=agent)
                else:
                    fn(query)

        # Instance attribute (not a class method) so getattr(handler, "on_agent_start")
        # returns exactly this function -- with the right iscoroutinefunction() answer
        # for THIS fn -- instead of always resolving to one fixed class-level method.
        self.on_agent_start = on_agent_start


class Agent(AgentLoopMixin, BaseAgent):
    """
    Agent Agent — Reasoning + Acting.

    Parameters
    ----------
    llm : BaseLLM | any
        Any LLM wrapper with .invoke() and optionally .ainvoke().
        Works with OpenAIChatModel, OpenAIResponse, or any compatible object.
    verbose : bool
        Print step-by-step execution to stdout.
    full_output : bool
        Also print raw LLM responses (useful for debugging).
    memory : MemoryProtocol, optional
        Memory backend for storing conversation history.
    max_iterations : int
        Hard limit on the number of Thought → Action → Observation steps.
    max_execution_time : float, optional
        Wall-clock time limit in seconds. Agent stops when exceeded.
    tool_timeout : float, optional
        Per-tool-call timeout in seconds. If a single tool call runs longer
        than this, it's abandoned and treated as an error Observation
        instead of blocking the agent loop forever -- max_execution_time
        alone can't catch this, since it's only checked between iterations,
        not while a tool call is in flight. A timed-out sync tool's thread
        keeps running in the background (Python can't force-stop a thread);
        an async tool is actually cancelled at its next await point.
        None (default) disables the timeout, matching prior behavior.
    max_scratchpad_tokens : int, optional
        Extra scratchpad budget on top of MAX_SCRATCHPAD_CHARS, measured in
        tokens instead of characters. Character count alone is a poor proxy
        for what actually overflows an LLM's context window -- tokens per
        character varies a lot by language and content (dense non-English
        text or code often runs well under the ~4 chars/token used for the
        default estimate). None (default) disables this and only the
        character cap applies, matching prior behavior.
    token_counter : callable, optional
        `fn(text: str) -> int` used to count tokens when max_scratchpad_tokens
        is set. Defaults to a rough len(text) // 4 approximation. Pass a real
        tokenizer for accuracy, e.g.
        `token_counter=lambda t: len(tiktoken.encoding_for_model(model).encode(t))`.
        Ignored if max_scratchpad_tokens is None.
    llm_retries : int
        Number of retries on a failed LLM call (rate limit, network blip,
        etc.) before giving up and raising AgentLLMError. 0 (default) means
        a single unconditional call, matching prior behavior -- no retries.
    llm_retry_backoff : float
        Base delay in seconds between retries. Backs off exponentially
        (backoff * 2**attempt), capped at llm_retry_max_backoff. Ignored if
        llm_retries is 0.
    llm_retry_max_backoff : float
        Upper bound in seconds on the exponential backoff delay.
    llm_retry_on : callable, optional
        `fn(exc: Exception) -> bool` deciding whether a given failure is
        worth retrying. Defaults to retrying everything except
        NotImplementedError (which signals tool_calling_mode="native" isn't
        supported by this LLM at all -- a config error, not transient).
    approval_callback : callable, optional
        Called before each tool execution as approval_callback(tool_name, tool_input).
        Return a truthy value to allow, falsy to deny.
    middleware : list[CallbackHandler], optional
        Event hooks for agent lifecycle events.
    max_consecutive_parse_errors : int
        Number of consecutive JSON parse failures before giving up.
    tools : list[dict], optional
        Initial tool list. More can be added with add_tools().
    toolbox : list[Toolbox], optional
        Toolboxes to lazy-load into this agent -- groups of tools that stay
        hidden from the initial prompt (only each toolbox's name/description
        is shown) until the agent calls `expose_toolbox(name)` or
        `expose_tool(tool_name)` at runtime. Keeps the context window clean
        when you have many tools but a given run only needs a few. Built
        internally on the same middleware mechanism as `middleware=`
        (adds a ToolboxMiddleware instance), so it composes with any other
        middleware you also pass. See autourgos_core.Toolbox.
    system_prompt : str
        Extra system-level instruction prepended to all requests.
    tool_calling_mode : "prompt" | "native"
        "prompt" (default): the original agent loop -- a plain-text prompt
        asks the model for a JSON {thought, actions, final_answer} object,
        parsed with a regex-based JSON extractor.
        "native": uses the LLM's own invoke_with_tools()/ainvoke_with_tools()
        (OpenAIChatModel, OpenAIResponse) -- structured tool_calls straight
        from the API, no text-JSON parsing, and multiple tool calls in one
        turn run concurrently. Raises if the given llm doesn't implement
        invoke_with_tools()/ainvoke_with_tools() (it defaults to raising
        NotImplementedError on BaseLLM). Conversation state is a real
        multi-turn message list rather than a single scratchpad string; the
        model's reasoning text isn't available when it also calls tools in
        the same turn (the wrapper doesn't currently return both), so
        Thought callbacks/logging are only fired on the final answer.
    max_scratchpad_chars : int, optional
        Per-instance override of MAX_SCRATCHPAD_CHARS (class default 15,000).
        Also used as the built-in summarizer's char-threshold trigger when
        ``summarize_every`` is set (see below) -- both share this one value.
    summarize_every : int, optional
        Enables built-in scratchpad summarization -- summarize every N
        iterations (and/or once the scratchpad exceeds
        ``max_scratchpad_chars``), using ``summarizer_llm`` if given, else
        this agent's own ``llm``. None (default) leaves summarization
        disabled -- the plain char-trim (``max_scratchpad_chars``) is all
        that applies, matching prior behavior. See README's
        "Auto-Summarizing Scratchpad" section.
    summarizer_llm : any with .invoke(), optional
        Dedicated LLM the built-in summarizer uses instead of this agent's
        own ``llm`` -- e.g. a cheaper/faster model just for compression.
        Only takes effect when ``summarize_every`` is also set; ignored
        otherwise.
    max_tool_output_chars : int, optional
        Per-instance override of MAX_TOOL_OUTPUT_CHARS (class default 5,000).
    max_tool_workers : int, optional
        Per-instance override of MAX_TOOL_WORKERS (class default 8).
    history : str, optional
        Folder path. When set, every run is recorded to a Markdown + JSON
        file pair under this folder (``Task_<timestamp>_<uid>.md``/``.json``)
        -- thoughts, tool calls, observations, and the final answer, with
        secret-shaped values (API keys, bearer tokens, JWTs, ...) redacted
        before writing. Written directly from the agent loop, not via
        `middleware=`. None (default) disables history recording entirely.
    on_agent_start : callable, optional
        Shortcut for the common case of wanting one function to run every
        time this agent starts (invoke()/ainvoke()), without writing a full
        CallbackHandler subclass. Called as fn(query) or, if it accepts it,
        fn(query, agent=self) -- same signature convention as
        CallbackHandler.on_agent_start. May be a plain function or an
        `async def`; both work from invoke() and ainvoke() (see
        CallbackManager's class docstring for how sync/async hooks are
        bridged). Internally just registers a CallbackHandler wrapping this
        function via add_middleware(), so it fires alongside (in the order
        added, after) any handlers passed via `middleware=`. Equivalent to:

            class _Start(CallbackHandler):
                def on_agent_start(self, query, agent=None, **kw):
                    fn(query)
            agent.add_middleware(_Start())
    pre_iteration_callback : callable, optional
        Sync or async ``callable(iteration: int)`` run before every
        iteration -- take a screenshot, refresh a cache, ping a health
        endpoint, etc. Written directly from the agent loop, not via
        `middleware=` (this only ever applies to the one Agent instance
        it's configured on).
    pre_iteration_files : str, list of str, or callable(iteration), optional
        File path(s) to inject into the LLM at every iteration. Pass a
        callable to generate paths dynamically (e.g. a screenshot that
        changes every iteration).
    image_quality : str or int
        Controls screenshot token cost when `pre_iteration_files` resolves
        to an image: ``"auto"`` (default, no change), ``"high"`` (no
        resize, forces `detail="high"`), ``"medium"`` (downscale to
        <=768px, JPEG q70), ``"low"`` (downscale to <=512px, JPEG q60), or
        an ``int`` 1-100 (JPEG quality directly). Ignored when
        `pre_iteration_files` is not set.
    """

    MAX_CONSECUTIVE_PARSE_ERRORS: int = 3
    MAX_SCRATCHPAD_CHARS:         int = 15_000
    MAX_TOOL_OUTPUT_CHARS:        int = 5_000
    MAX_TOOL_WORKERS:             int = 8

    def __init__(
        self,
        llm: Optional[Any] = None,
        verbose: bool = False,
        full_output: bool = False,
        memory: Optional[MemoryProtocol] = None,
        max_iterations: int = 15,
        max_execution_time: Optional[float] = None,
        tool_timeout: Optional[float] = None,
        max_scratchpad_tokens: Optional[int] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        llm_retries: int = 0,
        llm_retry_backoff: float = 1.0,
        llm_retry_max_backoff: float = 30.0,
        llm_retry_on: Optional[Callable[[BaseException], bool]] = None,
        approval_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        middleware: Optional[List[CallbackHandler]] = None,
        max_consecutive_parse_errors: int = 3,
        tools: Optional[List[Any]] = None,
        toolbox: Optional[List[Toolbox]] = None,
        system_prompt: str = "",
        tool_calling_mode: str = "prompt",
        max_scratchpad_chars: Optional[int] = None,
        summarize_every: Optional[int] = None,
        summarizer_llm: Optional[Any] = None,
        max_tool_output_chars: Optional[int] = None,
        max_tool_workers: Optional[int] = None,
        on_agent_start: Optional[Callable[..., Any]] = None,
        history: Optional[str] = None,
        pre_iteration_callback: Optional[Callable[[int], Any]] = None,
        pre_iteration_files: Optional[Union[str, List[str], Callable[[int], Any]]] = None,
        image_quality: Union[str, int] = "auto",
    ) -> None:
        if tool_calling_mode not in ("prompt", "native"):
            raise ValueError(
                f"tool_calling_mode must be 'prompt' or 'native', got {tool_calling_mode!r}."
            )
        self.tool_calling_mode = tool_calling_mode
        super().__init__(
            llm=llm,
            memory=memory,
            verbose=verbose,
            max_iterations=max_iterations,
            max_execution_time=max_execution_time,
            middleware=middleware,
            tools=tools,
        )
        self.full_output  = full_output
        self.tool_timeout = tool_timeout
        self.max_scratchpad_tokens = max_scratchpad_tokens
        self.token_counter = token_counter
        self.llm_retries = llm_retries
        self.llm_retry_backoff = llm_retry_backoff
        self.llm_retry_max_backoff = llm_retry_max_backoff
        self.llm_retry_on = llm_retry_on
        self.approval_callback = approval_callback
        self.max_consecutive_parse_errors = max_consecutive_parse_errors
        self.system_prompt   = system_prompt
        self.prompt_template = PREFIX_PROMPT + LOGIC_PROMPT + SUFFIX_PROMPT
        if max_scratchpad_chars is not None:
            self.MAX_SCRATCHPAD_CHARS = max_scratchpad_chars
        # Built-in scratchpad summarization -- inline in the loop (see
        # AgentLoopMixin._maybe_summarize/_amaybe_summarize in base.py),
        # NOT implemented via the CallbackHandler/middleware mechanism:
        # this only ever applies to the one Agent instance it's configured
        # on, so it needs none of a middleware's cross-instance-sharing
        # machinery (per-agent locks/registries).
        self.summarize_every = summarize_every
        self.summarizer_llm = summarizer_llm
        self._summarizer_lock = threading.Lock()
        self._last_summarized_length: Optional[int] = None
        self._warned_native_summarize = False
        if max_tool_output_chars is not None:
            self.MAX_TOOL_OUTPUT_CHARS = max_tool_output_chars
        if max_tool_workers is not None:
            self.MAX_TOOL_WORKERS = max_tool_workers
        self.logger = AgentLogger(
            verbose=verbose,
            agent_name="Agent",
            full_output=full_output,
        )
        if on_agent_start is not None:
            self.add_middleware(_FunctionStartHandler(on_agent_start))
        if toolbox:
            self.add_middleware(ToolboxMiddleware(toolboxes=toolbox))
        self._history = _HistoryRecorder(folder=history) if history else _NULL_HISTORY
        # Inline (non-middleware) pre-iteration file/callback injection --
        # native features of this package are plain constructor kwargs,
        # not middleware; the CallbackHandler/middleware bus is reserved
        # for third-party extensions (autourgos-hcix, autourgos-skills,
        # your own code).
        self._preiteration = (
            _PreIterationRuntime(pre_iteration_callback, pre_iteration_files, image_quality)
            if (pre_iteration_callback is not None or pre_iteration_files is not None)
            else _NULL_PREITERATION
        )

    # ── response parser ────────────────────────────────────────────────────────

    def _parser(self, response: str) -> tuple[Any, list, Any]:
        """
        Parse the LLM response into (thought, actions, final_answer).

        Expects a JSON object with keys:
            thought      — str | None
            actions      — list of {action, action_input} dicts
            final_answer — str | None
        """
        text = response if isinstance(response, str) else response.get("response", "")
        parsed = parse_json_object(text)

        thought      = parsed.get("thought")
        actions      = parsed.get("actions")
        final_answer = parsed.get("final_answer")

        # Normalise sentinel strings to Python None / []
        if thought in (None, "None", "null", ""):
            thought = None
        if not actions or actions in ("None", "null", ""):
            actions = []
        # `actions` must be a list of {action, action_input} dicts. A model
        # that replies with a single object instead of a one-item list (or
        # any other malformed shape) must be treated the same as any other
        # malformed response -- fed back through the parse-error retry path
        # -- rather than crashing the loop when base.py iterates it expecting
        # dicts (e.g. `for action_dict in actions: action_dict.get(...)`).
        elif not (isinstance(actions, list) and all(isinstance(a, dict) for a in actions)):
            actions = []
        if final_answer in (None, "None", "null", ""):
            final_answer = None

        return thought, actions, final_answer

    # ── public interface ──────────────────────────────────────────────────────

    def invoke(self, query: str, max_iterations: Optional[int] = None, **kwargs: Any) -> str:
        """
        Run the agent synchronously and return the final answer.

        Parameters
        ----------
        query : str
            The user's question or task.
        max_iterations : int, optional
            Override the instance-level max_iterations for this call.

        Returns
        -------
        str
            Final answer, or an error/timeout message prefixed with [Tag].
        """
        if not self.llm:
            raise ValueError("No LLM provided. Pass llm= to Agent().")

        with self._run_lock:
            if self._run_active:
                raise AgentAlreadyRunningError()
            self._run_active = True

        self.current_query = query
        resolved_max_iterations = (
            self.max_iterations if max_iterations is None else max_iterations
        )

        try:
            if self.memory:
                self.memory.add_user_message(query)
                self.logger.memory_action("Added user message to memory.")

            self.callback_manager.fire_agent_start(query, agent=self)
            self._history.start(query, agent_name=self.__class__.__name__)
            self.logger.run_start(query)

            if self.tool_calling_mode == "native":
                return self._run_loop_native(
                    query,
                    max_iterations=resolved_max_iterations,
                    approval_callback=self.approval_callback,
                    extra_kwargs=kwargs,
                )
            return self._run_loop(
                query,
                max_iterations=resolved_max_iterations,
                approval_callback=self.approval_callback,
                extra_kwargs=kwargs,
            )
        except BaseException as exc:
            # BaseException (not Exception) so a KeyboardInterrupt/SystemExit
            # mid-run still fires on_agent_error -- middleware (skills,
            # toolbox, hcix, history, preiteration) all do their cleanup
            # (removing injected tools/prompt blocks, stopping listeners,
            # flushing logs, deleting temp files) exclusively there. Bare
            # `raise` re-propagates it completely unchanged.
            self.callback_manager.fire_agent_error(exc, agent=self)
            self._history.fail(exc)
            self._preiteration.cleanup()
            raise
        finally:
            with self._run_lock:
                self._run_active = False
            self.logger.run_end()

    async def ainvoke(self, query: str, max_iterations: Optional[int] = None, **kwargs: Any) -> str:
        """
        Run the agent asynchronously and return the final answer.

        Parameters
        ----------
        query : str
            The user's question or task.
        max_iterations : int, optional
            Override the instance-level max_iterations for this call.
        """
        if not self.llm:
            raise ValueError("No LLM provided. Pass llm= to Agent().")

        with self._run_lock:
            if self._run_active:
                raise AgentAlreadyRunningError()
            self._run_active = True

        self.current_query = query
        resolved_max_iterations = (
            self.max_iterations if max_iterations is None else max_iterations
        )

        try:
            if self.memory:
                self.memory.add_user_message(query)
                self.logger.memory_action("Added user message to memory.")

            # Capture this run's contextvars.Context ONCE, before the first
            # hook fires, so a sync hook offloaded to a worker thread
            # (CallbackManager._afire/afire_before_iteration) reuses the
            # same Context across every hook call for this run instead of
            # a fresh, disconnected copy each time -- required for a
            # ContextVar-scoped middleware (e.g. autourgos-history's
            # per-run state) to see its own earlier writes. See
            # CallbackManager.capture_run_context()'s docstring.
            self.callback_manager.capture_run_context()

            await self.callback_manager.afire_agent_start(query, agent=self)
            self._history.start(query, agent_name=self.__class__.__name__)
            self.logger.run_start(query)

            if self.tool_calling_mode == "native":
                return await self._arun_loop_native(
                    query,
                    max_iterations=resolved_max_iterations,
                    approval_callback=self.approval_callback,
                    extra_kwargs=kwargs,
                )
            return await self._arun_loop(
                query,
                max_iterations=resolved_max_iterations,
                approval_callback=self.approval_callback,
                extra_kwargs=kwargs,
            )
        except BaseException as exc:
            # BaseException (not Exception) so asyncio.CancelledError (a
            # BaseException subclass, not Exception, since Python 3.8) mid-run
            # still fires on_agent_error -- see invoke()'s identical comment.
            # Bare `raise` re-propagates cancellation completely unchanged.
            await self.callback_manager.afire_agent_error(exc, agent=self)
            self._history.fail(exc)
            self._preiteration.cleanup()
            raise
        finally:
            with self._run_lock:
                self._run_active = False
            self.logger.run_end()
