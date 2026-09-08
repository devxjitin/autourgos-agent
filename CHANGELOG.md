# Changelog

## [3.10.0] - 2026-09-08

- **Added:** `Agent(pre_iteration_callback=..., pre_iteration_files=..., image_quality=...)` -- run a callback and/or inject files (e.g. a fresh screenshot) before every iteration, built directly into the agent loop rather than via `middleware=`. Matches the `history=`/`summarize_every=` pattern: this only ever applies to the one `Agent` instance it's configured on, so it needs none of a middleware's cross-instance-sharing machinery. The async path offloads to a worker thread (`run_in_executor`) so a slow callback/image-preprocess never blocks the event loop. See README's [Pre-Iteration Files & Callbacks](README.md#pre-iteration-files--callbacks).
- **Removed:** `PreIterationMiddleware` (added when this package absorbed the retired standalone `autourgos-preiteration` package). Superseded entirely by the inline kwargs above -- built-in features of this package are plain constructor kwargs, not middleware; the `CallbackHandler`/`middleware=` bus is reserved for third-party extensions (`autourgos-hcix`, `autourgos-skills`, your own code). **Breaking change** for anyone using `from autourgos_agent import PreIterationMiddleware` / `middleware=[PreIterationMiddleware(...)]` -- switch to `Agent(pre_iteration_callback=..., pre_iteration_files=..., image_quality=...)`. `SEQUENTIAL`/`PARALLEL` (for combining multiple callbacks) are unaffected and now used directly with `pre_iteration_callback=`.

## [3.7.0] - 2026-09-08

- **Added:** `Agent(history=<folder>)` -- inbuilt run history recording, replacing the separate `autourgos-history` package/middleware (now removed). When set, every run is written directly to a Markdown + JSON file pair under that folder (thoughts, tool calls, observations, final answer), with secret-shaped values (API keys, bearer tokens, JWTs, ...) redacted before writing. Implemented as a direct call from the agent loop (`autourgos_agent/history.py`'s `_HistoryRecorder`), not as a `CallbackHandler`/`middleware=` entry -- no extra package or wiring required. `None` (default) disables it entirely with zero overhead (`_NullHistory` no-op).

## [3.6.0] - 2026-09-08

- **Added:** `Agent.pause(reason=None)` / `Agent.resume()` / `Agent.is_paused` -- an in-process, thread-safe way to block a running agent at its next iteration boundary (never mid-tool-call or mid-LLM-call) until externally resumed. Works from both `invoke()`/`ainvoke()` and both `tool_calling_mode="prompt"`/`"native"`. Callable from any thread, not just the one running the agent; calling `pause()` before `invoke()`/`ainvoke()` starts means that run begins already paused. `resume()` without a prior `pause()` is a no-op. See README's [Pause & Resume](README.md#pause--resume).
- **Added:** Two new `CallbackHandler` hooks, `on_agent_pause(iteration, reason, agent=None, **kwargs)` and `on_agent_resume(iteration, paused_duration, agent=None, **kwargs)` (13 hooks total now), both with no-op defaults -- existing middleware is unaffected. Wired into `CallbackManager` as `fire_agent_pause`/`afire_agent_pause`/`fire_agent_resume`/`afire_agent_resume`.
- **Added:** `max_execution_time`'s deadline check now excludes time spent paused (`_elapsed_excluding_pauses()`), matching the same `total_paused_time` exclusion `autourgos-hcix`'s `CognitiveInterruptManager` already does for its own interrupt flow -- pausing an agent never counts against its execution-time budget.
- Internal: the async loops (`_arun_loop`, `_arun_loop_native`) block on the same `threading.Event` the sync loops use, offloaded to a worker thread via `run_in_executor` rather than a separate `asyncio.Event` -- avoids cross-thread thread-safety concerns since `pause()`/`resume()` must be callable from any thread.
- **Removed:** `AutoSummarizeMiddleware` and the `autourgos_agent.summarizer` module (added in 3.5.0, below) -- superseded by inline built-in summarization (next bullet) before ever reaching a published release beyond 3.5.0. **Breaking change** for anyone who adopted `from autourgos_agent import AutoSummarizeMiddleware` / `middleware=[AutoSummarizeMiddleware(...)]` during the brief 3.5.0 window -- use `Agent(summarize_every=..., summarizer_llm=..., max_scratchpad_chars=...)` instead.
- **Added:** Built-in scratchpad summarization via three `Agent()` constructor kwargs -- `summarize_every` (summarize every N iterations and/or once `max_scratchpad_chars` is exceeded), `summarizer_llm` (dedicated LLM for summarization; falls back to the agent's own `llm` when omitted), and the existing `max_scratchpad_chars` (now doing double duty as both the trim cap and the summarizer's char threshold, sharing one value). `summarize_every=None` (default) leaves summarization fully disabled, matching prior behavior. Implemented **inline in the agent loop** (`AgentLoopMixin._maybe_summarize`/`_amaybe_summarize`/`_do_summarize` in base.py) rather than as a `CallbackHandler`/middleware -- since it only ever applies to the one `Agent` instance it's configured on, it needs none of a middleware's cross-instance-sharing machinery (per-agent locks/registries collapse to plain instance state: one `threading.Lock`, one `_last_summarized_length`).
- Internal: `autourgos-core` floor reverted to `>=0.6.0` (was bumped to `>=0.7.0` in 3.5.0 for `PerAgentRegistry`/`warn_once_per_agent`, which the now-removed middleware needed and the inline replacement doesn't).

## [3.5.0] - 2026-09-08

- **Added:** `AutoSummarizeMiddleware`, merged in from the now-retired standalone `autourgos-summarizer` package as a built-in submodule (`autourgos_agent.summarizer`, exported from the package root). Periodically compresses `agent.scratchpad` to prevent token-window overflow (`summarize_every` iterations and/or `max_scratchpad_chars`), with per-agent locking, native-tool-calling-mode detection (scratchpad isn't sent to the LLM there, so summarizing it is skipped with a one-time warning), and an optional dedicated `llm=` for cheaper summarization. Behavior is unchanged from `autourgos-summarizer` 3.1.9 -- this is a location move, not a rewrite. Bumped `autourgos-core>=0.7.0` (the floor `autourgos-summarizer` already required for `PerAgentRegistry`/`warn_once_per_agent`).
- Install `autourgos-agent` alone now gets you the summarizer too -- no separate `autourgos-summarizer` package/install needed. `from autourgos_agent import Agent, AutoSummarizeMiddleware`.

## [3.4.0] - 2026-09-06

- **Added:** `CallbackHandler.on_iteration` may now return a truthy value to signal that the current prompt-mode iteration's action batch should be discarded instead of dispatched -- e.g. because the handler just injected a newer human instruction (`autourgos-hcix`) and the reasoning behind those actions is now stale. `CallbackManager.fire_iteration`/`afire_iteration` now aggregate and return this signal (OR semantics across handlers) instead of discarding every handler's return value; both loops (`_run_loop`, `_arun_loop`) skip the actions-dispatch block and append a scratchpad note when signaled, without incrementing `consecutive_parse_errors`. Not supported in `tool_calling_mode="native"` (no equivalent hook point there today). A handler returning `None`/falsy (every existing in-repo middleware) is completely unaffected.
- **Fixed:** `on_iteration` previously only fired when the LLM response also produced a non-empty `thought`, so a thought-less turn's action batch could never be discarded via this hook. Now fires every iteration, before the actions-dispatch block, regardless of `thought`.

## [3.3.0] - 2026-09-06

- **Fixed:** `Agent.invoke()`/`ainvoke()` had no guard against being called concurrently on the same instance -- a second overlapping call silently clobbered shared mid-run state (`current_query`, `scratchpad`) read by middleware during the run. Now raises the new `AgentAlreadyRunningError` instead; sequential (non-overlapping) reuse of one instance is unaffected. Use a separate `Agent` instance for concurrent work.
- **Fixed:** `max_tool_workers`/`MAX_TOOL_WORKERS` had no effect on the async tool-execution path (`ainvoke()`'s `asyncio.gather` ran every approved tool call with no cap at all) -- only the sync/`ThreadPoolExecutor` path enforced it. Async tool calls are now bounded by an `asyncio.Semaphore` sized to the same cap.
- **Fixed:** the sync loop created a brand-new `ThreadPoolExecutor` every iteration that had approved tool calls, so `MAX_TOOL_WORKERS` only ever capped one iteration's tool calls, not total in-flight threads across a multi-iteration run (each abandoned timed-out-tool thread from a prior iteration could pile up alongside a fresh full-size pool). Now one bounded pool is created per run and reused across iterations, shut down once at the end. Note: a tool that hangs past `tool_timeout` now permanently occupies one of that run's `MAX_TOOL_WORKERS` slots for the rest of the run (Python cannot force-stop a thread) -- a deliberate tradeoff for the cap actually holding, not a hidden regression.

## [3.2.0] - 2026-09-05

- **Fixed:** `CallbackManager._call_with_agent_fallback` (and the 3 duplicated `on_before_iteration`/`on_after_iteration` call sites) decided whether a handler accepts `agent=` by calling it and retrying on `TypeError` -- so a handler whose own body raised an unrelated `TypeError` got silently called a SECOND time, doubling any real side effect. Now decided up front via signature inspection (`_accepts_agent_kwarg`).
- **Fixed:** sync hooks offloaded to `CallbackManager`'s worker thread under `ainvoke()` didn't propagate `contextvars.ContextVar` writes between hook calls in the same run (a fresh `copy_context()` per call is a disconnected snapshot). Added `capture_run_context()` (called once per run by `Agent.ainvoke()`), reusing one `contextvars.Context` object for the whole run so ContextVar-scoped middleware state (e.g. autourgos-history's per-run state) is visible across hook calls.
- **Fixed:** `AgentLogger` had no `.warning()` method despite being duck-typed as a logger-shaped object by middleware (e.g. autourgos-hcix) that calls `.warning()` on it -- crashed with `AttributeError`, silently swallowed since it ran from inside a hook.
- Added: `inject_prompt_block()`/`remove_prompt_block()` in `runtime.py`, exported from the package root. Shared primitive for middleware that prepends/appends text into `agent.system_prompt`/`prompt_template` at runtime and needs to undo exactly that insertion later -- order-independent across multiple middleware, unlike a whole-string snapshot/restore (which is order-dependent and leaks text when middleware register/act in overlapping runs). Used by autourgos-toolbox, candidate for autourgos-skills/autourgos-hcix too.
- Regression coverage added for all of the above, plus a new assertion on `OpenAIResponse` native tool-calling that the Responses API's own `function_call`/`function_call_output` item shapes are sent (never a Chat-Completions-shaped `"role": "tool"`/`"tool_calls"` message).

## [3.1.6] - 2026-09-05

- `_tool_name()` gains a bare-callable fallback (`__name__`) for tools with neither a `"name"` key nor a `.name` attribute -- so `autourgos-toolbox` can import and share this function instead of reimplementing an equivalent lookup for its own unnormalized (`StructuredTool`/raw-callable) tool lists. Additive only; agent's own tool lists are always pre-normalized dicts and never hit the new branch.

## [3.1.5] - 2026-09-04

- Internal: `_call_llm_with_retry()`/`_acall_llm_with_retry()` now delegate to `autourgos_core.retry_with_backoff()`/`aretry_with_backoff()` (bumped `autourgos-core>=0.6.0`). No functional change -- backoff formula, `llm_retries`/`llm_retry_backoff`/`llm_retry_max_backoff`/`llm_retry_on` semantics, and logging all preserved (hand-verified attempt-by-attempt against the old loop, plus a new dedicated `tests/test_llm_retry.py` since none existed before). Live-verified against real Azure.

## [3.1.4] - 2026-09-04

- Internal: `_extract_text()` and `tool.py`'s `_parse_docstring_param_descriptions()` now delegate to `autourgos_core.extract_text()`/`parse_param_descriptions()` (bumped `autourgos-core>=0.4.0`). No functional change for this package.

## [3.1.3] - 2026-09-04

- Internal: `__version__` resolution moved to `autourgos_core.package_version()` (new `autourgos-core>=0.3.0` dependency). "No autourgos-core dependency" in the README referred to the old, since-removed v3 "typed vocabulary" `autourgos-core` package -- the new `autourgos-core` is a separate, zero-dependency stdlib utility library; README wording updated to clarify "no *third-party* dependencies". No functional change.

## [3.1.2] - 2026-09-03

- Added `features.md` documenting the module's feature set and a competitor comparison. No code changes.


## 3.1.1

- Metadata: added `maintainers` (Sonia, Vishwanil Suman) to `pyproject.toml`,
  and linked the README's existing Sonia maintainer badge to her GitHub
  profile (https://github.com/dahiyasonia). No code changes.

## 3.1.0

- Fixed: a malformed `actions` shape in the LLM response (e.g. a single
  object instead of a list of `{action, action_input}` dicts) crashed the
  loop with an uncaught `AttributeError` instead of being treated as a
  parse error like every other malformed-JSON case.
- Fixed: `invoke(query, max_iterations=0)`/`ainvoke(..., max_iterations=0)`
  silently fell back to the instance default instead of honoring the
  explicit override (`max_iterations or self.max_iterations` treated `0`
  as falsy).
- Added: `CallbackManager` now supports async hooks (`async def` handlers)
  from both `invoke()` and `ainvoke()`. From the async loop, a sync hook
  now runs off the event-loop thread instead of inline, so a blocking call
  inside it no longer stalls other concurrent `ainvoke()` runs sharing that
  thread.
- Changed: hook exceptions swallowed by `CallbackManager` now log at
  `WARNING` instead of `DEBUG`, so a buggy middleware handler is visible at
  default log levels.

## 3.0.0

- **Breaking:** removed the `backend="kernel"` bridge and everything it
  depended on -- `Agent(backend=..., capabilities=..., policy_executor_factory=...,
  max_effects=...)` constructor params, `kernel_backend.py`, and the soft
  re-export of `autourgos-core`'s typed vocabulary from the package's
  `__init__.py`. The v3 kernel/policy/capabilities stack this bridged to has
  been removed from the workspace; `autourgos-agent` returns to its original
  zero-dependency design with only `backend="legacy"` (the default and only
  loop implementation, unchanged).
- **Breaking:** removed `describe=`/`capability=`/`risk=` from `@tool` and
  `Tool` -- these existed solely to feed the removed policy pipeline. Plain
  tool dicts and `@tool`-decorated functions are unaffected otherwise.
- Removed the `core`/`kernel`/`policy` optional-dependency extras from
  `pyproject.toml`.
- No change to `backend="legacy"` behavior (async tool handling, native-mode
  system prompt rebuild, native message trimming, tool-less turns, or the
  `max_scratchpad_chars=`/`max_tool_output_chars=`/`max_tool_workers=`
  constructor params).

## 2.7.1

- **Security fix:** `Agent(backend="kernel", capabilities=[...])` now
  raises `ValueError` at construction time if `policy_executor_factory=`
  is not also given, instead of silently accepting the configuration and
  running every capability tool unguarded on first `invoke()` (matches
  `autourgos-kernel` 0.2.1's `Engine.run()` fail-closed guard, but fails
  earlier and with a clearer message since it's checked at construction).
- Fixed `tests/test_kernel_backend.py` importing `autourgos_core` and
  `autourgos_policy` (both optional extras) above their
  `pytest.importorskip()` guards, which turned a missing `autourgos-core`
  install into a hard collection error for the entire test run
  ("Interrupted: 1 error during collection", zero tests reported --
  not even the unrelated 95 `backend="legacy"` tests in other files)
  instead of a graceful skip of just that file. Requires `autourgos-kernel>=0.2.1`.

## 2.7.0

- Added kernel-only `capabilities`, `policy_executor_factory`, and
  `max_effects` options. A fresh policy executor is created for every run.
- Kernel tool calls can now flow through typed Action description, policy
  decision, confirmation, sandbox execution, and effect journaling.
- Extended `@tool` with opt-in `describe`, `capability`, and `risk` metadata;
  tools without those arguments keep their original four-key dictionary shape.
- Existing `approval_callback(tool_name, arguments)` semantics are preserved in
  policy mode and are invoked exactly once for every otherwise-allowed action.
- Updated optional dependency floors for core 0.2, kernel 0.2, and policy 0.3;
  the default install and legacy backend still have zero required dependencies.

## 2.6.0

- Added `Agent(backend="kernel")` (default remains `backend="legacy"`,
  the original loop implementation, completely unaffected): an opt-in
  bridge that delegates `invoke()`/`ainvoke()` to `autourgos-kernel`'s
  `Engine`/`Run` instead, for `Run`-based state isolation and
  checkpoint/resume. Requires `pip install autourgos-agent[kernel]`;
  `autourgos-kernel` is not a required dependency. Translates the
  kernel's exception hierarchy onto the existing `AgentError` subclasses
  and its event stream onto the existing `CallbackHandler` hooks and
  `agent.scratchpad`, so code written against `Agent`'s public contract
  doesn't need to know which backend ran. Has a few documented behavioral
  gaps vs `backend="legacy"` — see `autourgos_agent/kernel_backend.py`'s
  module docstring (no mid-run tool exposure, no `llm_retries`, no
  `on_iteration`/`on_before_iteration` hooks, `**kwargs` per-call LLM
  overrides not forwarded). 10 new tests (skipped, not failed, when
  `autourgos-kernel` isn't installed); the existing 95 tests are
  unaffected either way — verified both with and without
  `autourgos-core`/`autourgos-kernel` installed.

## 2.5.0

- Added an optional `[core]` extra (`pip install autourgos-agent[core]`)
  and a soft re-export of `autourgos-core`'s typed vocabulary (`Message`,
  `RunState`, `Budget`, `ArtifactRef`, `Action`, `Resource`, `Decision`,
  `Effect`, `Risk`, `ModelResponse`, and `CoreToolCall`/`CoreToolResult`/
  `CoreToolSpec`) -- only resolves if `autourgos-core` is separately
  installed, mirroring the `autourgos-memory` family's soft re-export
  pattern. `autourgos-core` is deliberately **not** a required dependency:
  this package's zero-required-dependencies design is unchanged, and
  nothing about `Agent`'s behavior depends on it being present.

## 2.4.0

- Fixed `async def` tools silently not running under `invoke()` (sync) --
  `_execute_tool()` never awaited a coroutine result, so an async tool's
  observation was the unawaited coroutine's `repr()` instead of its actual
  return value, with no error raised. `_execute_tool_async()` (used by
  `ainvoke()`) already handled this correctly; `_execute_tool()` now does too.
- Fixed `tool_calling_mode="native"` never re-reading `agent.system_prompt`
  after the loop started, so mid-run edits to it (e.g. `autourgos-hcix`'s
  human-override injection, `autourgos-toolbox`'s tool-unlock notice) were
  silently invisible to the model in native mode, even though the same
  mutation works as expected in the default `"prompt"` mode. The system
  message is now rebuilt from the live `agent.system_prompt` every iteration.
- Fixed `tool_calling_mode="native"` having no context-window budget at all
  -- the `messages` list grew unboundedly across iterations with nothing
  trimming it, unlike prompt mode's `MAX_SCRATCHPAD_CHARS`-bounded
  scratchpad. It's now trimmed to the same budget (plus
  `max_scratchpad_tokens` if set), dropping whole oldest turns so a
  `tool_call`/`tool` pairing is never split.
- `invoke()`/`ainvoke()` no longer require at least one tool -- a
  `ValueError("No tools added")` previously made tool-less turns (planning,
  clarification, plain conversation) impossible.
- Added `max_scratchpad_chars=`, `max_tool_output_chars=`, and
  `max_tool_workers=` constructor parameters on `Agent`, overriding the
  `MAX_SCRATCHPAD_CHARS` / `MAX_TOOL_OUTPUT_CHARS` / `MAX_TOOL_WORKERS`
  class defaults per-instance instead of requiring a subclass.

## 2.3.0

- Added `llm_retries`, `llm_retry_backoff`, `llm_retry_max_backoff`, and
  `llm_retry_on` (defaults `0`, `1.0`, `30.0`, `None`): automatic retry with
  exponential backoff on a failed LLM call, in place of raising
  `AgentLLMError` on the very first transient failure (rate limit, network
  blip, transient 5xx). Applies to `invoke()`/`ainvoke()` and both
  `tool_calling_mode` values. Defaults to retrying every exception except
  `NotImplementedError` (a config error signaling `tool_calling_mode="native"`
  isn't supported by the given LLM at all, not a transient one) — override
  with `llm_retry_on=fn(exc) -> bool` for finer control (e.g. only retry
  rate-limit errors). `llm_retries=0` (the default) makes this a single
  unconditional call, identical to prior behavior.

## 2.2.0

- Added `max_scratchpad_tokens` and `token_counter` (both default `None`): a
  token-based scratchpad budget on top of the existing character-based
  `MAX_SCRATCHPAD_CHARS`. Character count alone is a poor proxy for what
  actually overflows an LLM's context window — tokens per character varies a
  lot by language and content, so dense non-English text or code can blow a
  char-only budget's whole point long before the character cap trips.
  `token_counter` lets you plug in a real tokenizer (e.g. `tiktoken`); it
  defaults to a `len(text) // 4` approximation when not set. Both are no-ops
  unless `max_scratchpad_tokens` is set, matching prior behavior.

## 2.1.0

- Added `tool_timeout` (seconds, default `None`): a per-tool-call timeout that
  `max_execution_time` couldn't provide on its own, since that guard is only
  checked between loop iterations — a single hanging tool call (e.g. a
  network request with no timeout of its own) could block the agent loop
  forever regardless of `max_execution_time`. A timed-out call now becomes an
  error Observation (`"Error: tool '<name>' timed out after <n>s."`) and
  fires `on_tool_error`, instead of hanging. Applies in both `invoke()` and
  `ainvoke()`, and in both `tool_calling_mode="prompt"` and `"native"`.
- Fixed: the sync loops' `with ThreadPoolExecutor(...) as pool:` block was
  itself blocking on exit — `ThreadPoolExecutor.__exit__` calls
  `shutdown(wait=True)`, which waits for every submitted thread to finish,
  including one a tool-call timeout had already given up on. Replaced with
  explicit `pool.shutdown(wait=False)` so a timed-out tool call actually lets
  the loop move on immediately instead of blocking for the same duration
  anyway once the pool went out of scope.

## 2.0.2

- Fixed `tool_calling_mode="prompt"` (`_run_loop`/`_arun_loop`) not firing `on_tool_end` when `approval_callback` denies a tool call — only `on_tool_start` fired, so start/end-pairing middleware (metrics, tracing spans) never saw the call close. `tool_calling_mode="native"` already fired both; prompt mode now matches it.
- `approval_callback` passed to `invoke()` (sync) as an `async def` used to be silently treated as always-approved — calling it returns an unawaited coroutine, which is truthy — so every tool ran regardless of what the callback actually decided. It now raises a `TypeError` immediately, naming the problem and pointing at `ainvoke()` as the fix.
- Duck-typed tool objects (attributes instead of dict keys) now work end-to-end in `tool_calling_mode="prompt"`: `build_tool_list()`, the prompt-mode `tool_map`, and tool execution previously assumed a dict (`tool["name"]`, `tool.get("func")`) even though `tool_calling_mode="native"` already duck-typed the name lookup — a non-dict tool crashed before the native/prompt gap made it partially usable. Added `func`/`function` attribute lookup alongside the existing name lookup so a plain object with `.name`/`.description`/`.parameters`/`.func` works the same as a dict tool in both modes.
- `invoke()`'s per-step `ThreadPoolExecutor` for parallel tool calls is now capped at `Agent.MAX_TOOL_WORKERS` (default `8`) instead of spawning one thread per approved call — a step with many independent tool calls no longer opens unbounded threads at once.

## 2.0.1

- README: fixed the provider list (was an ASCII-art box) to a markdown table, added Maintainer badges (Sonia, Vishwanil Suman), and filled in the missing tool definitions (`weather_tool`/`calculator_tool`) in the Native Tool Calling example so it's actually copy-paste runnable.

## 2.0.0

**Breaking:**

- `invoke()`/`ainvoke()` now **raise** on a loop stop-condition instead of returning a `"[Tag] message"` string: `AgentTimeoutError`, `AgentMaxIterationsError`, `AgentParseError`, `AgentLLMError` (prompt mode), and `AgentEmptyResponseError` (native mode) — all subclasses of the new `AgentError`. Code that checked `result.startswith("[Max Iterations]")` etc. must now catch the corresponding exception instead. All five are exported from `autourgos_agent`.
- Fixed a bug in the parse-error counter (prompt mode): `consecutive_parse_errors` was reset to 0 on every non-throwing `_parser()` call, including one that returned no actions (e.g. a genuinely non-JSON response, which `parse_json_object` tolerates by returning `{}` rather than raising) — so the counter could never reach `max_consecutive_parse_errors` and a parse-error stop condition could never actually fire. It now only resets once a turn produces actions or a final answer.

**Other:**

- prompt-mode tool execution (`tool_calling_mode="prompt"`, the default) now runs approved tool calls from the same turn concurrently (`ThreadPoolExecutor` for `invoke()`, `asyncio.gather` for `ainvoke()`), matching what `prompt.py` already told the model was possible ("You can call multiple tools at once if they don't depend on each other's outputs") — previously only `tool_calling_mode="native"` actually ran them concurrently; prompt mode ran them one at a time regardless of what the prompt promised.
- `approval_callback` may now be an `async def` in `ainvoke()` (both prompt and native modes) — it's awaited if it returns an awaitable, and a plain sync callback still works unchanged in both `invoke()` and `ainvoke()`.
- Fixed `__version__` always falling back to the hardcoded `"2.0.0"` default — it looked up the PyPI distribution name `autourgos-react-agent`, left over from the fork, instead of `autourgos-agent`.

## 1.8.0

- `on_llm_end` now receives the raw LLM response and, when available, usage/cost/latency metadata (`provider_used`, `input_tokens`, `output_tokens`, `total_cost`, `latency_ms`, or `total_tokens` from a native SDK response's `.usage`) as callback kwargs, instead of only the extracted text. A cost-tracking middleware no longer needs to reach into `agent.llm`'s internal attributes.
- `add_tools()` now replaces an earlier tool on a name collision instead of keeping both registered. Previously both stayed listed in the LLM-facing tool prompt (with possibly contradictory descriptions) while only the most recently added implementation ever executed — a warning is still logged, but the prompt now matches what actually runs.
- Tool arguments are now validated against the tool function's signature before it's called. A wrong or missing argument from the model returns a message naming the expected signature instead of a raw Python `TypeError` surfacing as the tool's Observation.

## 1.7.1

Initial release of `autourgos-agent` — forked from `autourgos-react-agent`, generalized so the public API (`Agent`, `Create_Agent`) is no longer tied to ReAct-specific naming. Behavior is otherwise identical: same dual `tool_calling_mode` ("prompt" / "native"), same middleware/callback contracts, same tool decorator.

See [autourgos-react-agent's CHANGELOG](https://github.com/devxjitin/autourgos-react-agent/blob/main/CHANGELOG.md) for the full history prior to this fork.
