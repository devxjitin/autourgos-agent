# autourgos-agent — Features

A self-contained, general-purpose **ReAct-style LLM agent** for the Autourgos framework. It alternates Thought → Action → Observation until it produces a Final Answer. Zero required dependencies beyond Python 3.10+ — plug in any LLM wrapper exposing `.invoke()`/`.ainvoke()` (built to pair naturally with `autourgos-openaichat`/`autourgos-responses`, but works with anything OpenAI-compatible: Groq, Together AI, Mistral, DeepSeek, Perplexity, Ollama, LM Studio, vLLM).

## Full Feature List

### Core loop
- Sync (`invoke`) and async (`ainvoke`) ReAct-style Thought/Action/Observation loop
- Two tool-calling modes: `"prompt"` (JSON-in-text, the original loop) and `"native"` (the LLM's own structured tool calling via `invoke_with_tools`/`ainvoke_with_tools`, no regex JSON parsing)
- Parallel tool calls in one step — `ThreadPoolExecutor` (capped at `MAX_TOOL_WORKERS`, default 8) for `invoke()`, `asyncio.gather` for `ainvoke()`
- Live `agent.scratchpad` and `agent.current_query` instance attributes, readable mid-run by middleware
- Verbose colorized trace (`verbose=True`), plus raw-LLM-JSON dump (`full_output=True`)

### Tools
- `@tool` decorator — infers name/description/JSON-Schema parameters from type hints and a Google-style docstring; decorated function stays directly callable
- Plain-dict tool spec (`name`/`description`/`parameters`/`func`) for manual control
- Duck-typed tool objects (any object with `.name`/`.description`/`.parameters`/`.func`) — works in both tool-calling modes
- Sync and async tool functions supported in the same agent

### Reliability & limits
- `max_iterations` and `max_execution_time` hard limits, each raising a distinct typed exception (`AgentMaxIterationsError`, `AgentTimeoutError`)
- Per-tool-call `tool_timeout` — abandons a hanging tool call without blocking the loop forever (async tools are actually cancelled; sync tools' threads just get abandoned)
- Scratchpad size limits — char cap (`MAX_SCRATCHPAD_CHARS`, 15,000) plus an optional token-based cap (`max_scratchpad_tokens`, pluggable `token_counter`)
- LLM call retries with exponential backoff (`llm_retries`, `llm_retry_backoff`, `llm_retry_max_backoff`, customizable `llm_retry_on` predicate)
- `max_consecutive_parse_errors` — stops after repeated malformed-JSON responses instead of looping forever
- Typed exception hierarchy (`AgentError` base): `AgentTimeoutError`, `AgentMaxIterationsError`, `AgentParseError`, `AgentEmptyResponseError`, `AgentLLMError`

### Human-in-the-loop
- `approval_callback` gate before every tool execution — sync or async, denial produces a paired `on_tool_start`/`on_tool_end` with a "denied" observation rather than a silent skip

### Middleware / observability
- 11 lifecycle hooks via `CallbackHandler`: `on_agent_start/end/error`, `on_tool_start/end/error`, `on_iteration_start`, `on_before_iteration`, `on_iteration`, `on_llm_end`, `on_parse_error`
- Hooks may be sync or async regardless of whether the loop itself is sync or async
- `on_before_iteration` can inject/override kwargs (e.g. temperature) for a single iteration only
- `on_llm_end` surfaces cost/token/latency metadata from `autourgos-openaichat`/`-responses` for cost-tracking middleware
- `agent.logger.middleware(source, message)` — lets sibling middleware packages narrate their own actions into the same verbose trace

### Memory & lifecycle
- Pluggable `MemoryProtocol` for cross-call conversation history (integrates with the `autourgos-memory` family)
- Sync and async context-manager support (auto-closes the LLM's HTTP client)
- Custom persistent `system_prompt`

### Testing
- `autourgos_agent.testing.make_test_agent()` — builds a real `Agent` wired to a scripted fake LLM, zero network calls, for use in consumer test suites

### Compatibility
- `Create_Agent` legacy factory function retained with a `DeprecationWarning`

## Competitor Comparison

Landscape research on Python agent-orchestration frameworks, current as of the search date.

| Capability | **autourgos-agent** | Raw ReAct loop (hand-rolled) | [LangChain AgentExecutor / LangGraph](https://www.langchain.com/langgraph) | [CrewAI](https://www.crewai.com/) | [AutoGen](https://microsoft.github.io/autogen/) |
|---|---|---|---|---|---|
| Scope | In-process Python library, no separate service | N/A | Framework, graph/state-machine orchestration | Framework, role-based "crew" of agents | Framework, event-driven multi-agent messaging |
| Dependencies | Zero required beyond stdlib | N/A | Heavy — LangChain core + many integration packages | Heavy — LangChain-adjacent stack | Moderate — `autogen-core`/`autogen-agentchat` |
| Setup effort for first working agent | Low — single `Agent` + `@tool` | Low but you own everything | Moderate-high (80–150 LOC typical) | Low-moderate (30–60 LOC typical) | Moderate |
| ReAct Thought/Action/Observation loop | Yes, first-class, both prompt-JSON and native-tool-calling modes | Yes, if you write it | Yes (`create_react_agent`) plus much more (graphs, cycles) | Not the core model — role/task delegation instead | Not the core model — conversational turn-taking instead |
| Parallel tool calls in one step | Yes, built-in (`ThreadPoolExecutor`/`asyncio.gather`) | No, unless hand-built | Yes, via graph fan-out | Limited — sequential task execution is the default model | Yes, via concurrent agent messaging |
| Human-in-the-loop tool approval | Yes, built-in `approval_callback`, sync or async | No, unless hand-built | Yes, via `interrupt()`/checkpoint resume | Limited | Limited, via custom handlers |
| Hard iteration/time/tool-call limits | Yes, all three, each a typed exception | No, unless hand-built | Partial (recursion limits; no built-in wall-clock/tool timeout primitive) | Partial (max iterations per crew/task) | Partial (max turns) |
| LLM-call retry with backoff | Yes, built-in, configurable predicate | No, unless hand-built | Via LangChain's own retry wrappers | Via underlying LLM client only | Via underlying LLM client only |
| Middleware/callback hook surface | Yes, 11 typed hooks, sync+async, documented as a stable integration contract for sibling packages | No | Yes, via LangChain callbacks/tracers | Limited (crew/task callbacks) | Yes, via event/message hooks |
| Built-in test fixture for consumers | Yes (`make_test_agent`) | No | No official equivalent | No official equivalent | No official equivalent |
| Cost/usage passthrough to middleware | Yes, via `on_llm_end` kwargs | No | Via LangSmith (external service) | Via LangSmith/LiteLLM integrations | Via logging integrations |
| Provider lock-in | None — any `.invoke()`-shaped LLM object | None | Low, but ecosystem-heavy | Low-moderate | Low-moderate |
| Pricing | Free, open source | Free | Free, open source (LangSmith is paid) | Free, open source (CrewAI+ hosted tier paid) | Free, open source |

### How to read this

- **vs. a hand-rolled ReAct loop**: this is what most teams end up building by hand anyway (thought/action parsing, retries, timeouts, approval gates) — autourgos-agent packages it once, tested, with a stable middleware contract, instead of every project reinventing it slightly differently.
- **vs. LangChain/LangGraph**: LangGraph is far more powerful for genuinely stateful, cyclic, multi-actor workflows (durable checkpoints, time-travel, conditional branching) but at real setup and dependency cost; autourgos-agent stays a single, dependency-free library for the common single-agent ReAct case.
- **vs. CrewAI**: CrewAI's strength is fast-to-build multi-agent "teams" with roles and delegated tasks — a different mental model from a single tool-using ReAct agent, and it leans on the LangChain ecosystem underneath.
- **vs. AutoGen**: AutoGen is built around conversational multi-agent orchestration (debate, group chat) rather than a single agent looping over tools; it's the right tool when the unit of work is agent-to-agent conversation, not tool-calling.
- **General trade-off**: autourgos-agent trades multi-agent orchestration breadth for a minimal, zero-dependency, fully-typed single-agent core with strong reliability primitives (retries, timeouts, approval gates) built directly into the loop rather than bolted on.

Sources:
- [10 AI Agent Frameworks You Should Know in 2026: LangGraph, CrewAI, AutoGen & More](https://medium.com/@atnoforgenai/10-ai-agent-frameworks-you-should-know-in-2026-langgraph-crewai-autogen-more-2e0be4055556)
- [Autogen vs LangChain vs CrewAI: Our AI Engineers' Ultimate Comparison Guide](https://www.instinctools.com/blog/autogen-vs-langchain-vs-crewai/)
- [AI Agent Frameworks for Developers: LangChain vs CrewAI vs AutoGen in 2026](https://fungies.io/ai-agent-frameworks-langchain-crewai-autogen-2026/)
- [CrewAI vs LangGraph vs AutoGen vs OpenAgents — Best AI Agent Framework (2026)](https://openagents.org/blog/posts/2026-02-23-open-source-ai-agent-frameworks-compared)
- [AI Agent Frameworks 2026: LangGraph vs CrewAI vs AutoGen](https://cordum.io/blog/ai-agent-frameworks-comparison)
- [AI Agent Framework Comparison for Production: LangChain vs CrewAI vs AutoGen vs Just Using the API](https://www.clawagora.com/en/blog/ai-agent-framework-comparison-langchain-crewai-autogen)
