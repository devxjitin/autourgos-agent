# Changelog

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
