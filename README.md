# autourgos-agent

[![Framework: Autourgos](https://img.shields.io/badge/Framework-Autourgos-orange.svg)](https://github.com/devxjitin)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/autourgos-agent/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/devxjitin/autourgos-agent/blob/main/LICENSE)
[![Author](https://img.shields.io/badge/Author-Jitin%20Kumar%20Sengar-blue.svg)](https://github.com/devxjitin)
[![Maintainer](https://img.shields.io/badge/Maintainer-Sonia-blueviolet.svg)](https://github.com/dahiyasonia)
[![Maintainer](https://img.shields.io/badge/Maintainer-Vishwanil%20Suman-blueviolet.svg)]()

> A self-contained, **general-purpose LLM agent** for the Autourgos framework — reasoning and acting in one clean loop, with any OpenAI-compatible LLM you already have.

The agent alternates between **Thought** (reasoning about what to do next) and **Action** (calling a tool), looping until it has enough information to give a **Final Answer**.

**Fully self-contained** — zero third-party runtime dependencies beyond Python 3.10+ (only `autourgos-core`, itself a zero-dependency stdlib utility library shared across the framework). No forced LLM SDK, no forced vector store, no hidden network calls at import time. Bring your own LLM wrapper.

| Area | What you get |
|---|---|
| Core loop | Thought → Action → Observation, JSON-driven; native structured tool-calling mode too; parallel tool calls (thread pool / `asyncio.gather`); async everywhere — every method has an `a`-prefixed twin |
| Built in, zero extra install | Lazy-loaded **toolboxes** for large tool catalogs; per-iteration **file/callback injection** (screenshots, live data); **run history** to disk with secrets auto-redacted; **pause/resume** an in-flight run from any thread; auto-summarizing scratchpad, retry-with-backoff, timeouts |
| Extensible | `CallbackHandler` middleware with 11 lifecycle hooks; sync or async hooks, mixed freely, from either loop; approval callbacks for human-in-the-loop / safety gates; works with `autourgos-hcix`, `autourgos-skills`, and any hand-written middleware |
| Any LLM | OpenAI, Groq, Together AI, Mistral, DeepSeek, Perplexity; Ollama / LM Studio / vLLM — fully local, no API key; anything with `.invoke()` / `.ainvoke()` — no lock-in |

---

## Why use this?

Almost every major LLM provider today exposes an **OpenAI-compatible API**. `autourgos-agent` was designed with this in mind — it works with **any LLM** that has `.invoke()` and `.ainvoke()` methods. One agent, any provider:

| Provider | Notes |
|---|---|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, ... |
| Groq | Llama 3, Mixtral, Gemma |
| Together AI | 100+ open-source models |
| Mistral AI | `mistral-large`, `codestral` |
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` |
| Perplexity | `sonar` — web-connected |
| Ollama | local models, no internet |
| LM Studio | local models, GUI-based |
| vLLM | self-hosted, high throughput |

You are not locked to a single provider.

**What does it do?**

The agent receives a task and a list of tools. It then iterates:
1. **Think** — what information do I need? which tool should I call?
2. **Act** — call the tool, get the result
3. **Observe** — add the result to the scratchpad, repeat

This continues until the agent has a final answer or hits the iteration/time limit.

---

## Table of Contents

**Getting started**
- [Install](#install)
- [Quick Start](#quick-start)
- [How the Agent Loop Works](#how-the-agent-loop-works)
- [Defining Tools](#defining-tools)
- [Works With Any LLM](#works-with-any-llm)

**Running the agent**
- [Async Agent](#async-agent)
- [Parallel Tool Calls](#parallel-tool-calls)
- [Native Tool Calling](#native-tool-calling)
- [Verbose Mode](#verbose-mode)
- [Memory](#memory)
- [Approval Callback](#approval-callback)

**Extending the agent**
- [Middleware / Callbacks](#middleware--callbacks)
  - [Middleware Integration Contract](#middleware-integration-contract)
- [`on_agent_start` Shortcut](#on_agent_start-shortcut)
- [Toolboxes (Lazy-Loaded Tool Groups)](#toolboxes-lazy-loaded-tool-groups)
- [Pre-Iteration Files & Callbacks](#pre-iteration-files--callbacks)
- [Run History](#run-history)
- [Pause & Resume](#pause--resume)

**Operating it in production**
- [Testing](#testing)
- [Context Manager](#context-manager)
- [Time and Iteration Limits](#time-and-iteration-limits)
- [Scratchpad Size Limits](#scratchpad-size-limits)
  - [Auto-Summarizing Scratchpad](#auto-summarizing-scratchpad)
- [LLM Call Retries](#llm-call-retries)
- [Custom System Prompt](#custom-system-prompt)

**Reference**
- [Constructor Reference](#constructor-reference)
- [Tool Dict Reference](#tool-dict-reference)
- [What the Agent Returns](#what-the-agent-returns)
- [Exceptions](#exceptions)
- [v1 Backward Compatibility](#v1-backward-compatibility)

---

## Install

```bash
pip install autourgos-agent
```

No required third-party runtime dependencies (only `autourgos-core`, zero-dep itself). Bring your own LLM wrapper:

```bash
pip install autourgos-openaichat   # Chat Completions API
# or
pip install autourgos-responses    # OpenAI Responses API
```

Requires Python 3.10+.

---

## Quick Start

```python
from autourgos_agent import Agent, tool
from autourgos_openaichat  import OpenAIChatModel

# 1. Define a tool
@tool
def calculator(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

# 2. Create the agent
agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    verbose=True,
)
agent.add_tools(calculator)

# 3. Run
result = agent.invoke("What is 123 + 456?")
print(result)
# 579
```

(Prefer writing tools by hand instead? A plain dict — `{"name", "description", "parameters", "func"}` — still works exactly the same way; see [Defining Tools](#defining-tools) below.)

Expected verbose output (LangChain-flavored Thought/Action/Observation trace):
```
> Starting Agent...

Thought: I need to add 123 and 456. I'll use the calculator tool.
Action: calculator
Action Input: {'a': 123, 'b': 456}
Observation: 579.0
Thought: I have the result from the calculator.
Final Answer: 123 + 456 = 579

> Agent finished.
```

(Shown here without colour; in a real terminal `> Starting Agent...`,
`> Agent finished.`, and `Final Answer:` print in bold green, `Action:` /
`Action Input:` in yellow, `Observation:` in blue/cyan, `Thought:` in cyan,
and `Parse Error:` in red.)

---

## How the Agent Loop Works

Each iteration the agent produces a JSON object:

```json
{
  "thought": "I need to search for the latest Python version.",
  "actions": [
    {"action": "search", "action_input": {"query": "latest Python version 2025"}}
  ],
  "final_answer": null
}
```

Rules the LLM must follow (enforced by the prompt):

- If tools are needed → fill `actions`, set `final_answer` to `null`
- If the answer is ready → fill `final_answer`, set `actions` to `[]`
- Never set both `actions` and `final_answer` at the same time
- Multiple tools can be called in one step if they are independent

The agent collects tool results into a **scratchpad** that is passed back to the LLM at each step so it always has full context of what was tried.

---

## Defining Tools

There are two ways to define a tool. Both produce the same shape under the
hood and can be freely mixed on the same agent.

### Recommended: the `@tool` decorator

Decorate a type-hinted function and the `name`, `description`, and
JSON-Schema `parameters` are inferred automatically — no dict to write by hand:

```python
from autourgos_agent import tool

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. Tokyo
        unit: celsius or fahrenheit
    """
    # Replace with real API call
    return f"The weather in {city} is 22°{unit[0].upper()} and sunny."

agent.add_tools(get_weather)
```

- `name` defaults to the function's name (override with `@tool(name=...)`).
- `description` defaults to the first line of the docstring (override with `@tool(description=...)`).
- `parameters` are inferred from type hints (`str`/`int`/`float`/`bool`/`list`/`dict` →
  the matching JSON-Schema type; parameters without a default are marked `required`).
  Per-parameter descriptions are parsed from a Google-style `Args:` section if present.
  Override entirely with `@tool(parameters={...})` if you need something the
  inference can't express.
- The decorated function stays directly callable — `get_weather("Tokyo")` still
  works outside the agent, e.g. in your own tests.

```python
# Overriding the inferred name/description:
@tool(name="calculator", description="Add two numbers together.")
def add(a: float, b: float) -> float:
    return a + b
```

### Alternative: a plain dict

For full manual control (or if you're integrating an existing tool spec),
a tool can still be a plain Python dict with these keys:

| Key | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | yes | Tool name (used by the LLM to call it) |
| `description` | `str` | yes | What the tool does — shown to the LLM |
| `parameters` | `dict` | recommended | JSON-Schema object describing the inputs |
| `func` | `callable` | yes | Python function to call |

```python
weather_tool = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string",  "description": "City name, e.g. Tokyo"},
            "unit": {"type": "string",  "description": "celsius or fahrenheit"},
        },
        "required": ["city"],
    },
    "func": get_weather,
}
```

### Duck-typed tool objects

A tool doesn't have to be a dict at all — any object exposing `.name`,
`.description`, `.parameters`, and `.func` (or `.function`) as attributes
instead of dict keys works the same way in both `tool_calling_mode="prompt"`
and `"native"`:

```python
class WeatherTool:
    name = "get_weather"
    description = "Get the current weather for a city."
    parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    func = staticmethod(get_weather)

agent.add_tools(WeatherTool())
```

Useful when wrapping an existing tool class from another library instead of
reshaping it into a dict.

### Adding tools

`@tool`-decorated functions and plain dicts both work the same way with
`add_tools` / the constructor, and can be mixed freely:

```python
# One at a time
agent.add_tools(get_weather)

# Multiple at once (mix @tool and dict tools freely)
agent.add_tools(get_weather, calculator_tool, search_tool)

# From a list
agent.add_tools([get_weather, calculator_tool])

# Via constructor
agent = Agent(llm=llm, tools=[get_weather, calculator_tool])
```

---

## Works With Any LLM

Change the `llm=` argument to switch providers. Everything else stays the same.

### OpenAI

```python
from autourgos_openaichat import OpenAIChatModel

agent = Agent(llm=OpenAIChatModel(model="gpt-4o", api_key="sk-..."))
```

### Groq (very fast, free tier)

```python
from autourgos_openaichat import OpenAIChatModel

agent = Agent(
    llm=OpenAIChatModel(
        model="llama3-70b-8192",
        api_key="gsk_...",
        base_url="https://api.groq.com/openai/v1",
    )
)
```

### Ollama (fully local, no internet, no API key)

```bash
ollama pull llama3
```

```python
from autourgos_openaichat import OpenAIChatModel

agent = Agent(
    llm=OpenAIChatModel(
        model="llama3",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )
)
```

### Together AI

```python
from autourgos_openaichat import OpenAIChatModel

agent = Agent(
    llm=OpenAIChatModel(
        model="meta-llama/Llama-3-70b-chat-hf",
        api_key="...",
        base_url="https://api.together.xyz/v1",
    )
)
```

### Mistral AI

```python
from autourgos_openaichat import OpenAIChatModel

agent = Agent(
    llm=OpenAIChatModel(
        model="mistral-large-latest",
        api_key="...",
        base_url="https://api.mistral.ai/v1",
    )
)
```

### DeepSeek

```python
from autourgos_openaichat import OpenAIChatModel

agent = Agent(
    llm=OpenAIChatModel(
        model="deepseek-chat",
        api_key="...",
        base_url="https://api.deepseek.com/v1",
    )
)
```

### OpenAI Responses API (autourgos-responses)

```python
from autourgos_responses import OpenAIResponse

agent = Agent(llm=OpenAIResponse(model="gpt-4o"))
```

### LM Studio (local GUI)

```python
from autourgos_openaichat import OpenAIChatModel

agent = Agent(
    llm=OpenAIChatModel(
        model="local-model",
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
    )
)
```

### vLLM (self-hosted)

```python
from autourgos_openaichat import OpenAIChatModel

agent = Agent(
    llm=OpenAIChatModel(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        api_key="EMPTY",
        base_url="http://your-server:8000/v1",
    )
)
```

---

## Async Agent

All agent methods have an async counterpart.

```python
import asyncio
from autourgos_agent import Agent
from autourgos_openaichat  import OpenAIChatModel

agent = Agent(llm=OpenAIChatModel(model="gpt-4o"))
agent.add_tools(weather_tool, calculator_tool)

async def main():
    result = await agent.ainvoke("What is the weather in Tokyo and what is 99 * 3?")
    print(result)
    # The weather in Tokyo is 22°C and sunny. 99 × 3 = 297.

asyncio.run(main())
```

Async tools (coroutine functions) are also supported:

```python
import httpx

@tool
async def search(query: str) -> str:
    """Search the web."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://api.example.com/search?q={query}")
        return r.text
# async function — works with ainvoke(), same as a plain-dict "func" would.
```

---

## Parallel Tool Calls

The LLM can call multiple tools in a single step when they don't depend on each other. The agent runs the approved ones concurrently (a `ThreadPoolExecutor` for `invoke()`, `asyncio.gather` for `ainvoke()`) and collects all results before the next LLM call. `invoke()`'s thread pool is capped at `Agent.MAX_TOOL_WORKERS` (default `8`) regardless of how many tool calls the model requests in one step, so a step with more approved calls than that queues the rest rather than spawning one thread per call.

```python
agent = Agent(llm=OpenAIChatModel(model="gpt-4o"))
agent.add_tools(weather_tool, calculator_tool, search_tool)

result = agent.invoke(
    "What is the weather in Paris and London, and what is 250 * 4?"
)
print(result)
# The weather in Paris is 18°C cloudy, London is 15°C rainy. 250 × 4 = 1000.
```

The LLM produces three tool calls in one step:
```json
{
  "thought": "I can fetch both cities' weather and compute the multiplication in parallel.",
  "actions": [
    {"action": "get_weather", "action_input": {"city": "Paris"}},
    {"action": "get_weather", "action_input": {"city": "London"}},
    {"action": "calculator",  "action_input": {"a": 250, "b": 4}}
  ],
  "final_answer": null
}
```

For native structured tool-calling instead of JSON-in-text parsing, see [Native Tool Calling](#native-tool-calling) below.

---

## Native Tool Calling

`tool_calling_mode="native"` swaps the JSON-in-text agent loop for the LLM's own structured tool-calling — `invoke_with_tools()`/`ainvoke_with_tools()` on `OpenAIChatModel`/`OpenAIResponse`. No regex JSON parsing — tool calls come back as structured data straight from the API. Multiple tool calls in one turn run concurrently here too, the same way `"prompt"` mode does.

```python
from autourgos_agent import Agent, tool
from autourgos_openaichat import OpenAIChatModel

@tool
def weather_tool(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: 18°C, cloudy"

@tool
def calculator_tool(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    tool_calling_mode="native",
)
agent.add_tools(weather_tool, calculator_tool)

result = agent.invoke("What is the weather in Paris, and what is 250 * 4?")
print(result)
```

Requires the LLM to implement `invoke_with_tools()`/`ainvoke_with_tools()` — both `OpenAIChatModel` and `OpenAIResponse` do. Passing `tool_calling_mode="native"` with an LLM that doesn't (a plain duck-typed `.invoke()`-only object, or the default `BaseLLM.invoke_with_tools()` stub) raises a `RuntimeError` immediately, naming the LLM class and what's missing — it never silently falls back to prompt mode.

Trade-offs versus `"prompt"` mode (the default):
- **More reliable**: no `AgentParseError` from malformed JSON, since the API returns structured tool calls directly.
- **No visible "Thought" per step**: when the model also calls tools, the wrapper doesn't currently return accompanying reasoning text alongside the tool calls, so `on_iteration`/`logger.thought()` aren't fired on tool-call turns — only on the final answer.
- Conversation state is a real multi-turn message list, not the single rendered scratchpad string `"prompt"` mode uses. `agent.scratchpad` is still kept up to date as a human-readable trace (for middleware that reads it), but it isn't what's actually sent to the LLM in this mode.

---

## Verbose Mode

Enable `verbose=True` to print every step to stdout.

```python
agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    verbose=True,
)
agent.add_tools(weather_tool)
result = agent.invoke("What is the weather in Sydney?")
```

Output (LangChain-flavored Thought/Action/Observation trace):
```
> Starting Agent...

Thought: I need to get the weather in Sydney using the get_weather tool.
Action: get_weather
Action Input: {'city': 'Sydney'}
Observation: The weather in Sydney is 25°C and sunny.
Thought: I have the weather information for Sydney.
Final Answer: The weather in Sydney is 25°C and sunny.

> Agent finished.
```

Enable `full_output=True` to also print the raw LLM JSON at each step — useful for debugging prompt or parse issues:

```python
agent = Agent(llm=llm, verbose=True, full_output=True)
```

---

## Memory

Attach a memory backend to persist conversation history across calls.

```python
from autourgos_agent import Agent, MemoryProtocol
from typing import Dict, List

class SimpleMemory(MemoryProtocol):
    def __init__(self):
        self._history: List[Dict[str, str]] = []

    def add_user_message(self, message: str) -> None:
        self._history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str) -> None:
        self._history.append({"role": "assistant", "content": message})

    def get_history(self) -> List[Dict[str, str]]:
        return list(self._history)

memory = SimpleMemory()
agent  = Agent(llm=llm, memory=memory)
agent.add_tools(search_tool)

result1 = agent.invoke("Search for the capital of France.")
print(result1)
# The capital of France is Paris.

result2 = agent.invoke("What city did I just ask about?")
print(result2)
# You asked about Paris, the capital of France.
```

---

## Approval Callback

Require human (or programmatic) approval before any tool is executed.

```python
def require_approval(tool_name: str, tool_input: dict) -> bool:
    print(f"\n[Approval required] Tool: {tool_name}")
    print(f"Input: {tool_input}")
    answer = input("Allow? (y/n): ").strip().lower()
    return answer == "y"

agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    approval_callback=require_approval,
)
agent.add_tools(delete_file_tool)
result = agent.invoke("Delete the temp folder.")
```

If the callback returns a falsy value, the tool is skipped and the agent sees:
```
Observation: Tool call was denied by the approval callback.
```

A denial still fires both `on_tool_start` and `on_tool_end` (with that same
"denied by the approval callback" message as the result) — middleware that
tracks tool calls by pairing start/end events sees a consistent pair either
way, never a start with no matching end.

Use this to implement human-in-the-loop, audit logging, or safety checks for destructive tools.

`approval_callback` can also be an `async def` — `ainvoke()` awaits it if it returns an awaitable (e.g. to wait on a Slack approval), and a plain sync callback still works unchanged in both `invoke()` and `ainvoke()`:

```python
async def require_approval(tool_name: str, tool_input: dict) -> bool:
    return await ask_on_slack(tool_name, tool_input)

agent = Agent(llm=OpenAIChatModel(model="gpt-4o"), approval_callback=require_approval)
agent.add_tools(delete_file_tool)
result = await agent.ainvoke("Delete the temp folder.")
```

Note: `invoke()` (the sync entrypoint) only supports a sync `approval_callback`.
Passing an `async def` callback to `invoke()` raises a `TypeError` right away
(naming the problem and pointing at the fix) instead of silently approving
every tool — call `agent.ainvoke()` instead, or use a plain sync callback.

---

## Middleware / Callbacks

Register event hooks to observe the agent without modifying it. `CallbackHandler` exposes 11 hooks in total: `on_agent_start`, `on_agent_end`, `on_agent_error`, `on_tool_start`, `on_tool_end`, `on_tool_error`, `on_iteration_start`, `on_before_iteration`, `on_iteration`, `on_llm_end`, and `on_parse_error`. Every hook may receive an `agent=<Agent instance>` kwarg (older handlers that don't accept it still work).

**`on_agent_error` also fires on cancellation.** `invoke()`/`ainvoke()` catch `BaseException`, not just `Exception`, so `on_agent_error` is guaranteed to fire — exactly once, with a bare re-raise afterward — for an ordinary exception, a cancelled async run (`asyncio.CancelledError`), or a `KeyboardInterrupt`/`SystemExit` mid-run. Do cleanup that must always happen (removing tools/prompt blocks your middleware injected, stopping a listener, flushing a log, deleting temp files) in `on_agent_error`, not just `on_agent_end` — otherwise a cancelled run silently skips it. If your handler treats `on_agent_error` as "a real application failure occurred" (e.g. alerting), check `isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit))` to distinguish cancellation from an actual error.

```python
from autourgos_agent import Agent, CallbackHandler

class MyLogger(CallbackHandler):

    def on_agent_start(self, query: str, **kwargs) -> None:
        print(f"Agent started with query: {query}")

    def on_agent_end(self, result: str, **kwargs) -> None:
        print(f"Agent finished: {result}")

    def on_tool_start(self, tool_name: str, tool_input: dict, **kwargs) -> None:
        print(f"Calling tool: {tool_name} with {tool_input}")

    def on_tool_end(self, tool_name: str, result: str, **kwargs) -> None:
        print(f"Tool {tool_name} returned: {result[:100]}")

    def on_iteration(self, iteration: int, thought: str, **kwargs) -> None:
        print(f"Iteration {iteration} — thought: {thought}")

    def on_parse_error(self, iteration: int, raw_response: str, **kwargs) -> None:
        print(f"Parse error at step {iteration}: {raw_response[:100]}")

    # Extra hooks (v1.1.0+) — all optional, all receive `agent=` too:
    def on_agent_error(self, error: Exception, **kwargs) -> None:
        print(f"Agent error: {error}")

    def on_tool_error(self, tool_name: str, error: Exception, **kwargs) -> None:
        print(f"Tool {tool_name} raised: {error}")

    def on_iteration_start(self, iteration: int, **kwargs) -> None:
        print(f"Starting iteration {iteration}")

    def on_llm_end(self, response, **kwargs) -> None:
        print(f"LLM responded: {str(response)[:100]}")
        # kwargs also carries raw=<untouched raw LLM response> plus, when the
        # wrapper exposes it: provider_used, input_tokens, output_tokens,
        # total_cost, latency_ms (autourgos-openaichat/-responses' dict shape)
        # or total_tokens (from a native SDK response's .usage) — useful for
        # cost/usage-tracking middleware without reaching into agent.llm.
        if "total_cost" in kwargs:
            print(f"  cost so far: ${kwargs['total_cost']}")


agent = Agent(llm=llm, middleware=[MyLogger()])
agent.add_tools(weather_tool)
result = agent.invoke("Weather in Berlin?")
```

You can also add middleware after construction:

```python
agent.add_middleware(MyLogger())
```

### Narrating middleware activity in the verbose trace

By default, `verbose=True` only shows the core loop (Thought/Action/Observation) --
middleware changing the agent's tools, scratchpad, or prompt behind the scenes is
otherwise invisible. Middleware can narrate what it's doing into the same trace via
`agent.logger.middleware(source, message)`:

```python
class MyLogger(CallbackHandler):
    def on_agent_start(self, query: str, agent=None, **kwargs) -> None:
        logger = getattr(agent, "logger", None)
        if logger:
            logger.middleware("MyLogger", "Doing something worth narrating.")
```

Printed in magenta with a `[Source]` prefix so it's unambiguous which middleware
produced the line, e.g.:

```
[Toolbox] Exposed toolbox 'search_tools' to agent.
[Summarizer] Compressed scratchpad (iteration 5, was 15,320 chars).
```

Use `getattr(agent, "logger", None)` (not a direct import of `AgentLogger`) so your
middleware doesn't crash if it's ever attached to something other than a `Agent`,
and does nothing when `verbose=False`. The built-in summarizer and pre-iteration
runtime (see [Auto-Summarizing Scratchpad](#auto-summarizing-scratchpad) and
[Pre-Iteration Files & Callbacks](#pre-iteration-files--callbacks) — both
implemented inline, not as middleware, but narrate the same way), the built-in
`ToolboxMiddleware` (see [Toolboxes](#toolboxes-lazy-loaded-tool-groups)), and
sibling middleware packages (e.g. `autourgos-hcix`, `autourgos-skills`) all use
this same pattern to narrate their own actions.

### Middleware Integration Contract

These are the three pieces of surface area sibling middleware (the
built-in `ToolboxMiddleware`, `autourgos-hcix`, `autourgos-skills`, and
anything else you write) can rely on. This is the official, stable
contract — treat it as public API.

**`agent.scratchpad` (str)**
A real, live instance attribute, not just a local loop variable. It is
updated in place on every iteration of `invoke()`/`ainvoke()`, so a
callback handler (or any other code holding a reference to the agent) can
read it *while the loop is still running*, not only after it finishes.
It's reset to `""` at the start of every `invoke()`/`ainvoke()` call, so
calling `invoke()` twice on the same agent instance never leaks the
previous run's scratchpad into the new one.

```python
class ScratchpadWatcher(CallbackHandler):
    def on_iteration_start(self, iteration, agent=None, **kwargs):
        print(f"[iter {iteration}] scratchpad so far:\n{agent.scratchpad}")
```

**`agent.current_query` (str)**
Set once, at the start of every `invoke()`/`ainvoke()` call, to the query
being worked on. Lets middleware answer "what is this agent doing right
now?" without threading the query through every hook signature.

**`on_tool_start` / `on_tool_end` are always paired**
Every `on_tool_start` for a given tool call is followed by exactly one
`on_tool_end` for that same call — including when `approval_callback`
denies it (see [Approval Callback](#approval-callback)). Middleware that
tracks in-flight tool calls (e.g. an "active calls" gauge, or a tracing
span opened on start and closed on end) can rely on this pairing without
special-casing denials.

**`on_before_iteration(iteration, agent=None, **kwargs)`**
Called once per loop iteration, right before the LLM is invoked for that
iteration. If a handler returns a `dict`, its keys are merged into the
`self.llm.invoke()` / `self.llm.ainvoke()` call for *that iteration only*
— it is not persisted to later iterations. If multiple handlers return
dicts, later handlers win on key conflicts. Returning `None` (the
default no-op, same as every other hook) changes nothing.

```python
class TemperatureOverride(CallbackHandler):
    def on_before_iteration(self, iteration, agent=None, **kwargs):
        # only lower the temperature on the first iteration
        if iteration == 1:
            return {"temperature": 0.0}
        return None
```

This is how middleware can, for example, inject a trace id, override a
sampling parameter, or attach per-call metadata without the agent needing
to know anything about the specific middleware doing it.

**Hooks may be sync or async, from either `invoke()` or `ainvoke()`**
Define a hook as a plain `def` or an `async def` — both work from both
loops:

- From `ainvoke()` (the async loop), a sync hook runs on a background
  thread rather than the event-loop thread, so a blocking call inside it
  (an LLM request, a file write, `time.sleep`, anything) doesn't stall the
  loop for every other concurrent `ainvoke()` call sharing that thread. An
  async hook is awaited directly.
- From `invoke()` (the sync loop), a sync hook is called directly, same as
  always; an async hook is driven to completion with its own
  short-lived event loop.

```python
class RemoteAudit(CallbackHandler):
    async def on_iteration_start(self, iteration, agent=None, **kwargs):
        await audit_client.log(agent.current_query, iteration)
```

You don't need to pick one style for a whole handler — different hooks on
the same class can mix sync and async freely.

---

## `on_agent_start` Shortcut

For the common case of wanting one function to run every time an agent
starts, without writing a full `CallbackHandler` subclass:

```python
def log_start(query: str) -> None:
    print(f"Starting: {query}")

agent = Agent(llm=llm, on_agent_start=log_start)
```

`on_agent_start` may be a plain function or an `async def` — both work from
`invoke()` and `ainvoke()`. It's called as `fn(query)`, or `fn(query, agent=self)`
if the function's signature accepts an `agent` kwarg:

```python
async def log_start(query: str, agent=None) -> None:
    await audit_client.log(query, agent.current_query)

agent = Agent(llm=llm, on_agent_start=log_start)
```

This is internally just sugar for `agent.add_middleware(...)` wrapping your
function in a `CallbackHandler` — it fires alongside (after) any handlers
passed via `middleware=`.

---

## Toolboxes (Lazy-Loaded Tool Groups)

`toolbox=` groups tools under a name + description that stay **hidden**
from the agent's prompt until it actually needs them — only each toolbox's
name and one-line description are shown upfront. The agent calls the
built-in `expose_toolbox(name)` (or `expose_tool(tool_name)` for a single
tool) meta-tool at runtime to load the real tools. Useful when you have
many tools across several domains but a given run only needs a handful —
keeps the context window clean instead of dumping every tool's full schema
into the prompt every time.

```python
from autourgos_core import tool, Toolbox
from autourgos_agent import Agent
from autourgos_openaichat import OpenAIChatModel

@tool
def web_search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

@tool
def scrape_url(url: str) -> str:
    """Scrape a web page."""
    return f"Scraped content of {url}"

@tool
def run_query(sql: str) -> str:
    """Run a SQL query against the database."""
    return f"Rows for: {sql}"

@tool
def list_tables() -> str:
    """List all tables in the database."""
    return "users, orders, products"

web = Toolbox(name="web", description="Web search and page scraping tools.",
              tools=[web_search, scrape_url])
db  = Toolbox(name="database", description="SQL query tools.",
              tools=[run_query, list_tables])

agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    toolbox=[web, db],
)

result = agent.invoke("Find the latest Python release and save it to the DB")
```

The agent first sees only:
```
## Dynamic Toolboxes
- **web**: Web search and page scraping tools.
- **database**: SQL query tools.
```
Then, when it decides it needs one, it calls `expose_toolbox("web")` (or
`expose_tool("web_search")` for just that one tool), which registers the
real tool(s) on the agent and injects their full schemas into the prompt
for the rest of that run. Everything toolbox-related — meta-tools, injected
schemas, displaced/restored tools — is automatically cleaned up at
`on_agent_end`/`on_agent_error`, so the agent returns to its original state
before the next `invoke()`.

You can also mix `tools=` (always visible) with `toolbox=` (lazy-loaded)
on the same agent:

```python
agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    tools=[calculator],       # always in the prompt
    toolbox=[web, db],        # hidden until expose_toolbox()/expose_tool() is called
)
```

`toolbox=` is sugar for `add_middleware(ToolboxMiddleware(toolboxes=[...]))`
under the hood — `ToolboxMiddleware` and `Toolbox` are both importable from
`autourgos_agent`/`autourgos_core` if you need to build one dynamically:

```python
from autourgos_agent import ToolboxMiddleware

mw = ToolboxMiddleware()
mw.add_toolbox("web", "Web search and page scraping tools.", [web_search, scrape_url])
agent.add_middleware(mw)
```

---

## Pre-Iteration Files & Callbacks

Run a callback and/or inject files (e.g. a fresh screenshot) before every
agent iteration — useful for computer-use / vision agents, live data feeds,
or anything that needs to refresh before each step.

Built directly into the agent loop via `pre_iteration_callback=`/
`pre_iteration_files=` on `Agent()` — same pattern as `history=`/
`summarize_every=`. Not middleware: no `CallbackHandler` to write, no
`middleware=[...]` to wire up, zero overhead when unset.

```python
from autourgos_agent import Agent
from autourgos_openaichat import OpenAIChatModel

SCREENSHOT = "/tmp/screen.png"

def capture(iteration: int) -> None:
    take_screenshot(SCREENSHOT)  # your own screenshot function

agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    pre_iteration_callback=capture,
    pre_iteration_files=SCREENSHOT,
    image_quality="low",   # downscale to <=512px, JPEG q60 -- ~85 tokens flat
)
result = agent.invoke("Click the 'Submit' button on screen.")
```

`pre_iteration_files=` also accepts a callable that returns a path (or list
of paths) for dynamic/per-iteration file names, and non-image files are
passed through unmodified:

```python
agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    pre_iteration_files=lambda iteration: f"/tmp/screen_{iteration}.png",
)
```

`image_quality` options: `"auto"` (default, no change), `"high"` (no
resize, `detail="high"`), `"medium"` (≤768px, JPEG q70), `"low"` (≤512px,
JPEG q60), or an `int` 1–100 (JPEG quality directly). Resizing requires
Pillow: `pip install 'autourgos-agent[images]'` — without it, only the
`detail=` hint is applied (still saves tokens on the OpenAI side) and the
original image is sent unresized.

The callback can be sync or async, and runs safely from both `invoke()`
(directly) and `ainvoke()` (offloaded to a worker thread so it never
blocks the event loop). A callback that raises is logged (at `ERROR`) and
does not stop the agent run.

Run multiple callbacks with `SEQUENTIAL` (one after another) or `PARALLEL`
(concurrently — sync callbacks in a thread pool, async ones as asyncio
tasks):

```python
from autourgos_agent import SEQUENTIAL, PARALLEL

def log_step(iteration: int) -> None:
    print(f"Iteration {iteration} starting")

async def refresh_cache(iteration: int) -> None:
    await cache.refresh()

agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    pre_iteration_callback=SEQUENTIAL[capture, log_step],      # capture, then log, in order
    # or: pre_iteration_callback=PARALLEL[capture, refresh_cache],  # both at once
    pre_iteration_files=SCREENSHOT,
)
```

---

## Run History

`history=` records every run to a Markdown + JSON file pair on disk —
thoughts, tool calls, observations, and the final answer — written
directly by the agent loop (not middleware), with secret-shaped values
(API keys, bearer tokens, JWTs, ...) automatically redacted before writing.

```python
agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    history="./agent_runs",   # folder is created if it doesn't exist
)
agent.add_tools(search_tool)

result = agent.invoke("Research the latest AI news.")
# writes ./agent_runs/Task_<timestamp>_<uid>.md and .json
```

The Markdown file is a human-readable transcript of the run (thought/action/
observation trace plus the final answer); the JSON file is the same data in
a structured form for programmatic use. `history=None` (the default)
disables recording entirely — no files are written.

---

## Pause & Resume

`agent.pause(reason=None)` / `agent.resume()` / `agent.is_paused` give you
an in-process, thread-safe way to pause a running agent and later hand
control back to it — the run blocks at its next **iteration boundary**
(before the next LLM call, never mid-tool-call or mid-LLM-call) until
`resume()` is called. Works from both `invoke()` and `ainvoke()`, and in
both `tool_calling_mode="prompt"` and `"native"`.

```python
agent = Agent(llm=my_llm)

# From another thread (or another asyncio task):
agent.pause(reason="waiting for human review")
...
agent.resume()
```

- **Call `pause()`/`resume()` from any thread** — they don't have to be
  called from the thread running `invoke()`/`ainvoke()`.
- **Calling `pause()` before `invoke()`/`ainvoke()` starts** means that run
  begins already paused — it blocks before its first iteration. This is a
  valid way to start an agent pre-paused, not a bug.
- **`resume()` without a prior `pause()` is a no-op.**
- **`max_execution_time` excludes time spent paused** — pausing an agent
  (e.g. to wait for a human) never counts against its execution-time
  budget.
- A middleware hook can pause the agent it's attached to just as easily as
  external code:

```python
class PauseForApproval(CallbackHandler):
    def on_iteration_start(self, iteration, agent=None, **kwargs):
        if needs_human_review(agent):
            agent.pause(reason="needs review")
```

Two new `CallbackHandler` hooks narrate pause/resume to middleware:

```python
class PauseLogger(CallbackHandler):
    def on_agent_pause(self, iteration, reason, agent=None, **kwargs):
        print(f"Paused at iteration {iteration}: {reason}")

    def on_agent_resume(self, iteration, paused_duration, agent=None, **kwargs):
        print(f"Resumed after {paused_duration:.1f}s")
```

Out of scope for this feature (deliberately): resuming a paused run in a
*different* process or after the original one has exited — this is an
in-process mechanism, not a serialized/checkpointed one. Pausing mid-tool-call
is also not supported — a pause only ever takes effect at the next
iteration boundary.

---

## Testing

`autourgos_agent.testing` ships `make_test_agent()` — a shared test
fixture that builds a real, fully-functional `Agent` wired to a
scripted fake LLM, with zero network calls. Use it in your own tests
instead of hand-rolling a fake agent (a hand-rolled fake's shape can
silently drift from the real `Agent` and hide real bugs):

```python
import json
from autourgos_agent.testing import make_test_agent

agent = make_test_agent(responses=[
    json.dumps({"thought": "thinking", "actions": [], "final_answer": "42"}),
])
result = agent.invoke("what is the answer?")
assert result == "42"
assert agent.llm.call_count == 1
```

`make_test_agent()` accepts `responses` (a list of raw JSON-text canned
LLM replies in the `{thought, actions, final_answer}` format), and
optional `tools`, `memory`, `middleware`, `max_iterations`, and any other
`Agent` constructor kwarg. If `tools` is omitted, a harmless `echo`
tool is attached automatically so `agent.invoke()` works out of the box.

---

## Context Manager

The agent implements both sync and async context managers. They automatically close the LLM's HTTP client when the block exits.

```python
with Agent(llm=OpenAIChatModel(model="gpt-4o")) as agent:
    agent.add_tools(calculator_tool)
    result = agent.invoke("What is 7 * 8?")
    print(result)
    # 56
# LLM client closed here
```

Async:

```python
import asyncio
from autourgos_openaichat import OpenAIChatModel

async def main():
    async with Agent(llm=OpenAIChatModel(model="gpt-4o")) as agent:
        agent.add_tools(calculator_tool)
        result = await agent.ainvoke("What is 12 ** 2?")
        print(result)
        # 144

asyncio.run(main())
```

---

## Time and Iteration Limits

Prevent runaway agents with hard limits.

```python
agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    max_iterations=10,       # stop after 10 Thought → Action → Observe cycles
    max_execution_time=30.0, # stop after 30 seconds wall-clock time
)
agent.add_tools(search_tool)

try:
    result = agent.invoke("Research the entire history of the internet.")
except AgentTimeoutError:
    ...  # 30s wall-clock elapsed without a final answer
except AgentMaxIterationsError:
    ...  # 10 iterations elapsed without a final answer
```

You can also override `max_iterations` per call:

```python
result = agent.invoke("Quick question: capital of Japan?", max_iterations=3)
```

`max_execution_time` is rechecked immediately after every blocking LLM call,
tool wait, and approval-callback call returns (not just once per iteration),
and an in-flight async LLM call is actually cancelled at its next await point
via `asyncio.wait_for`. It still can't force-stop a hanging *synchronous* call
already in progress — Python has no way to preempt a running sync frame — so
it detects an overrun as soon as possible rather than truly interrupting one.
Use `tool_timeout` to bound a single tool call instead:

```python
agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    tool_timeout=10.0,  # abandon any single tool call that runs past 10s
)
agent.add_tools(flaky_network_tool)

result = agent.invoke("Fetch the data and summarize it.")
# If flaky_network_tool hangs, its Observation becomes:
# "Error: tool 'flaky_network_tool' timed out after 10.0s."
# instead of blocking the agent loop forever.
```

A timed-out sync tool's underlying thread keeps running in the background
(Python has no way to force-stop a running thread) — the agent loop itself
just stops waiting on it. An async tool is actually cancelled at its next
`await` point. `tool_timeout=None` (the default) disables this and matches
prior behavior.

---

## Scratchpad Size Limits

The scratchpad (`agent.scratchpad`, `"prompt"` mode only) is capped at
`Agent.MAX_SCRATCHPAD_CHARS` (15,000 characters) by default — once exceeded,
older steps are trimmed from the front and replaced with
`"[...earlier steps trimmed...]"`.

Character count alone is a poor proxy for what actually overflows an LLM's
context window: tokens per character varies a lot by language and content
(dense non-English text or code can run well under the ~4 chars/token rule
of thumb, silently blowing a char-only budget's whole point long before
15,000 characters is reached). `max_scratchpad_tokens` adds a second,
token-based cap on top of the character one:

```python
agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    max_scratchpad_tokens=4000,  # trim further if the scratchpad exceeds ~4000 tokens
)
```

Without a real tokenizer, token count is approximated as `len(text) // 4`
(the common English-prose rule of thumb). Pass `token_counter=` for
precision — any `fn(text: str) -> int`, e.g.:

```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o")

agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    max_scratchpad_tokens=4000,
    token_counter=lambda text: len(encoding.encode(text)),
)
```

`max_scratchpad_tokens=None` (the default) disables the token-based check —
only the character cap applies, matching prior behavior.

### Auto-Summarizing Scratchpad

The blunt trim above (drop older steps) throws away content. For long-running
tool-heavy agents, built-in summarization periodically replaces
`agent.scratchpad` with an LLM-generated summary instead, preserving key
findings and tool results while shrinking the token footprint. It's
implemented inline in the agent loop itself (not as middleware), turned on
with three `Agent()` constructor kwargs:

```python
from autourgos_agent import Agent

agent = Agent(llm=my_llm, summarize_every=5, max_scratchpad_chars=8000)
```

`max_scratchpad_chars` does double duty here — it's both the trim cap
(`Agent.MAX_SCRATCHPAD_CHARS`) and the summarizer's own char-threshold
trigger, sharing this one value. `summarize_every=None` (the default) leaves
summarization disabled entirely, matching prior behavior.

Pass `summarizer_llm=` to use a separate, cheaper/faster model just for
summarization instead of this agent's own `llm` (ignored if `summarize_every`
isn't set):

```python
from autourgos_openaichat import OpenAIChatModel

cheap_llm = OpenAIChatModel(model="gpt-4o-mini")
agent = Agent(llm=my_llm, summarize_every=5, summarizer_llm=cheap_llm)
```

Notes:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `summarize_every` | `int \| None` | `None` | Summarize every N iterations. `None` disables summarization entirely. |
| `max_scratchpad_chars` | `int` | `15000` | Trim cap; also triggers summarization once the scratchpad exceeds it (when `summarize_every` is set). |
| `summarizer_llm` | any with `.invoke()` | `None` | Dedicated LLM for summarization; falls back to this agent's own `llm` when omitted. |

- Has no effect in `tool_calling_mode="native"` — `agent.scratchpad` is a
  human-readable trace only in that mode and isn't sent to the LLM, so
  summarizing it wouldn't shrink the real context-window budget. Skips,
  warning once per agent.
- A concurrent summarization attempt for the same agent (e.g. two overlapping
  calls somehow racing) skips rather than blocking.
- On success, narrates via `agent.logger.middleware("Summarizer", ...)` (see
  [Narrating middleware activity](#narrating-middleware-activity-in-the-verbose-trace)).

---

## LLM Call Retries

By default, any failed LLM call (rate limit, network blip, transient 5xx)
raises `AgentLLMError` immediately and ends the run — the same call would
often succeed a moment later. `llm_retries` retries with exponential
backoff instead:

```python
agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    llm_retries=3,             # retry up to 3 times before giving up
    llm_retry_backoff=1.0,     # base delay: 1s, 2s, 4s (capped below)
    llm_retry_max_backoff=30.0,
)
agent.add_tools(search_tool)

result = agent.invoke("What's the latest news?")
# A rate-limited call now retries instead of failing the whole run outright.
```

By default every exception is retried **except** `NotImplementedError`
(the signal that `tool_calling_mode="native"` isn't supported by this LLM at
all — a config error, not a transient one, so retrying it would just delay
the clearer error). Pass `llm_retry_on` to customize which errors are worth
retrying:

```python
def only_rate_limits(exc: Exception) -> bool:
    return "rate limit" in str(exc).lower()

agent = Agent(llm=llm, llm_retries=5, llm_retry_on=only_rate_limits)
```

`llm_retries=0` (the default) disables this and matches prior behavior — a
single unconditional call, raising `AgentLLMError` on the first failure.

---

## Custom System Prompt

Add extra instructions that persist across all steps.

```python
agent = Agent(
    llm=OpenAIChatModel(model="gpt-4o"),
    system_prompt=(
        "You are a helpful financial analyst. "
        "Always cite your sources. "
        "Never speculate without data."
    ),
)
agent.add_tools(search_tool, calculator_tool)
result = agent.invoke("What is the P/E ratio of Apple?")
```

---

## Constructor Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `llm` | any | `None` | LLM wrapper with `.invoke()` / `.ainvoke()`. Works with `OpenAIChatModel`, `OpenAIResponse`, or any compatible object |
| `verbose` | `bool` | `False` | Print step-by-step execution to stdout |
| `full_output` | `bool` | `False` | Also print raw LLM responses (implies `verbose`) |
| `memory` | `MemoryProtocol` | `None` | Memory backend for conversation history |
| `max_iterations` | `int` | `15` | Max Thought → Action → Observe cycles before stopping |
| `max_execution_time` | `float` | `None` | Wall-clock time limit in seconds |
| `tool_timeout` | `float` | `None` | Per-tool-call timeout in seconds. See [Time and Iteration Limits](#time-and-iteration-limits) |
| `max_scratchpad_tokens` | `int` | `None` | Extra token-based scratchpad budget on top of `MAX_SCRATCHPAD_CHARS`. See [Scratchpad Size Limits](#scratchpad-size-limits) |
| `token_counter` | `callable` | `None` | `fn(text) -> int` used to count tokens for `max_scratchpad_tokens`. Defaults to a `len(text) // 4` approximation |
| `llm_retries` | `int` | `0` | Retries on a failed LLM call before raising `AgentLLMError`. See [LLM Call Retries](#llm-call-retries) |
| `llm_retry_backoff` | `float` | `1.0` | Base delay in seconds between retries (exponential: `backoff * 2**attempt`) |
| `llm_retry_max_backoff` | `float` | `30.0` | Upper bound in seconds on the exponential backoff delay |
| `llm_retry_on` | `callable` | `None` | `fn(exc) -> bool` deciding whether a failure is worth retrying. Defaults to retrying everything except `NotImplementedError` |
| `approval_callback` | `callable` | `None` | Called as `fn(tool_name, tool_input)` before each tool. Return truthy to allow |
| `middleware` | `list[CallbackHandler]` | `None` | Event hooks for lifecycle events |
| `max_consecutive_parse_errors` | `int` | `3` | Stop after this many back-to-back JSON parse failures |
| `tools` | `list[dict]` | `None` | Initial tool list (more can be added with `add_tools()`) |
| `toolbox` | `list[Toolbox]` | `None` | Toolboxes to lazy-load — hidden from the prompt until `expose_toolbox()`/`expose_tool()` is called. See [Toolboxes](#toolboxes-lazy-loaded-tool-groups) |
| `system_prompt` | `str` | `""` | Extra system-level instruction added to every prompt |
| `tool_calling_mode` | `"prompt"` \| `"native"` | `"prompt"` | `"prompt"`: the original JSON-in-text agent loop. `"native"`: uses the LLM's `invoke_with_tools()`/`ainvoke_with_tools()` — structured tool calls straight from the API, no JSON parsing, and multiple tool calls in one turn run concurrently. See [Native Tool Calling](#native-tool-calling) |
| `max_scratchpad_chars` | `int` | `None` (class default 15,000) | Per-instance override of the scratchpad trim cap; also the built-in summarizer's char threshold when `summarize_every` is set. See [Scratchpad Size Limits](#scratchpad-size-limits) |
| `summarize_every` | `int` | `None` | Enables built-in scratchpad summarization every N iterations, using this agent's own `llm` (or `summarizer_llm`, if given). See [Auto-Summarizing Scratchpad](#auto-summarizing-scratchpad) |
| `summarizer_llm` | any with `.invoke()` | `None` | Dedicated LLM the built-in summarizer uses instead of this agent's own `llm`. Only takes effect when `summarize_every` is also set |
| `max_tool_output_chars` | `int` | `None` (class default 5,000) | Per-instance override of the max characters kept from a single tool's result before truncating with `"... [truncated]"` |
| `max_tool_workers` | `int` | `None` (class default 8) | Per-instance override of the thread-pool size `invoke()` uses to run parallel tool calls. See [Parallel Tool Calls](#parallel-tool-calls) |
| `on_agent_start` | `callable` | `None` | `fn(query)` (or `fn(query, agent=self)`) run every time this agent starts, without writing a full `CallbackHandler`. See [`on_agent_start` Shortcut](#on_agent_start-shortcut) |
| `history` | `str` | `None` | Folder path — records every run to a Markdown + JSON file pair, with secrets redacted. See [Run History](#run-history) |
| `pre_iteration_callback` | `callable` | `None` | Sync or async `fn(iteration)` run before every iteration. See [Pre-Iteration Files & Callbacks](#pre-iteration-files--callbacks) |
| `pre_iteration_files` | `str \| list[str] \| callable` | `None` | File path(s) (or a callable returning them) injected into the LLM call at every iteration |
| `image_quality` | `str \| int` | `"auto"` | Screenshot/image token-cost control for `pre_iteration_files`: `"auto"`, `"high"`, `"medium"`, `"low"`, or an `int` 1–100 JPEG quality. Ignored when `pre_iteration_files` isn't set |

---

## Tool Dict Reference

| Key | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | yes | Identifier used by the LLM. Use snake_case |
| `description` | `str` | yes | Plain-English description of what the tool does and when to use it |
| `parameters` | `dict` | recommended | JSON-Schema `object` describing the function's inputs |
| `func` | `callable` | yes | The Python function to call. Can be sync or async |

`parameters` format (JSON Schema):

```python
"parameters": {
    "type": "object",
    "properties": {
        "param_name": {
            "type": "string",       # string | number | integer | boolean | array | object
            "description": "...",   # shown to the LLM — make it clear
            "enum": ["a", "b"],     # optional: restrict to specific values
        },
    },
    "required": ["param_name"],     # list required params
}
```

---

## What the Agent Returns

- **Normal completion** — `invoke()`/`ainvoke()` return a `str`: the final answer extracted from the LLM's `final_answer` field
- **Error / limit reached** — raises one of the exceptions below instead of returning a string, so callers catch a type rather than string-sniffing the result

---

## Exceptions

All of these are exported from `autourgos_agent` and subclass `AgentError`:

| Exception | Meaning |
|---|---|
| `AgentTimeoutError` | `max_execution_time` was exceeded |
| `AgentMaxIterationsError` | `max_iterations` reached without a final answer |
| `AgentParseError` | `tool_calling_mode="prompt"`: LLM failed to produce valid JSON `max_consecutive_parse_errors` times in a row |
| `AgentEmptyResponseError` | `tool_calling_mode="native"`: LLM returned neither a final answer nor tool calls `max_consecutive_parse_errors` times in a row |
| `AgentLLMError` | LLM raised an exception (network, rate limit, etc.) — the original exception is on `.original` |

```python
from autourgos_agent import Agent, AgentError, AgentTimeoutError

try:
    result = agent.invoke("...")
except AgentTimeoutError:
    ...
except AgentError:
    ...  # catches any of the above
```

---

## v1 Backward Compatibility

The old `Create_Agent` factory function still works but emits a `DeprecationWarning`:

```python
from autourgos_agent import Create_Agent  # DeprecationWarning

agent = Create_Agent(llm=llm)  # same as Agent(llm=llm)
```

Update your code to use `Agent` directly.

---

## License

Apache License 2.0, Copyright (c) 2026 Jitin Kumar Sengar
