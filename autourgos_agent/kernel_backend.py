"""
kernel_backend.py — Agent(backend="kernel") bridge to autourgos-kernel.

Opt-in only. backend="legacy" (the default) is the original _run_loop /
_arun_loop / _run_loop_native / _arun_loop_native implementation in
base.py, completely untouched by this module -- existing code and tests
keep working exactly as before regardless of whether autourgos-kernel is
even installed.

backend="kernel" delegates invoke()/ainvoke() to autourgos_kernel's
Engine/Run instead, translating the kernel's exception hierarchy and
event stream onto Agent's existing public contract (AgentError subclasses,
CallbackHandler hooks, agent.scratchpad, memory=) so callers who want
Run-based concurrency isolation and checkpoint/resume can opt in without
changing anything else about how they use Agent.

Known gaps vs backend="legacy" (Phase 2, Part B of the Autourgos v3
roadmap -- see idea.md; these are documented limitations, not silently
ignored):

  - Tools added mid-run via agent.add_tools() from inside a tool call
    (autourgos-toolbox's expose_toolbox() pattern) are NOT picked up --
    the kernel Engine's tool index is built once per invoke() call from
    self.tools at call time, not rebuilt every iteration the way
    backend="legacy" does since its Phase 0 fix.
  - llm_retries / llm_retry_backoff / llm_retry_max_backoff / llm_retry_on
    are not applied -- autourgos-kernel has no LLM-call retry layer yet.
  - max_consecutive_parse_errors maps to the kernel's
    max_consecutive_empty, a related but not identical condition (an
    empty response vs an unparseable one); prompt mode's retry-prompt
    text also differs slightly from backend="legacy"'s.
  - on_iteration (the "thought" callback) never fires -- the kernel's
    ModelResponse doesn't carry reasoning text separately from tool_calls
    or the final answer in either tool-calling mode.
  - on_llm_end fires with the response text only, not the
    provider_used/token/cost metadata backend="legacy" extracts from the
    raw LLM response (autourgos_kernel's llm_end event doesn't carry it).
  - agent.scratchpad is reconstructed from the kernel's event stream for
    middleware compatibility, but its exact text may differ in minor
    formatting details from backend="legacy"'s.
  - on_before_iteration's per-iteration LLM-kwargs override return value
    is not applied -- autourgos_kernel's Engine has no equivalent hook.
  - invoke()/ainvoke()'s **kwargs (per-call LLM overrides, e.g.
    temperature=) are NOT forwarded -- autourgos_kernel.Engine.run()
    doesn't accept them yet.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from .base import (
    AgentEmptyResponseError,
    AgentError,
    AgentLLMError,
    AgentMaxIterationsError,
    AgentTimeoutError,
    CallbackManager,
    _get_memory_context,
    _record_agent_message,
)


def _import_kernel() -> Any:
    try:
        import autourgos_kernel
    except ImportError as exc:
        raise ImportError(
            "backend='kernel' requires autourgos-kernel. Install it with "
            "`pip install autourgos-agent[kernel]`."
        ) from exc
    return autourgos_kernel


def _translate_error(exc: BaseException, kernel: Any) -> BaseException:
    """Map an autourgos_kernel exception onto the matching AgentError
    subclass, so code written against Agent's existing exception contract
    (try/except AgentTimeoutError, etc.) doesn't need to know which
    backend actually ran."""
    if isinstance(exc, kernel.RunTimeoutError):
        return AgentTimeoutError(exc.max_execution_seconds)
    if isinstance(exc, kernel.MaxIterationsError):
        return AgentMaxIterationsError(exc.max_iterations)
    if isinstance(exc, kernel.EmptyResponseError):
        return AgentEmptyResponseError(exc.consecutive_empty)
    if isinstance(exc, kernel.ProviderError):
        return AgentLLMError(exc.original)
    if isinstance(exc, kernel.RunCancelledError):
        return AgentError(str(exc))
    return exc


def _render_scratchpad_step(iteration: int, tool_events: List[Dict[str, Any]]) -> str:
    lines = [f"\nStep {iteration}:"]
    for ev in tool_events:
        lines.append(f"Action: {ev['tool']}({ev.get('arguments', {})})")
        lines.append(f"Observation: {ev.get('content', '')}")
    return "\n".join(lines)


async def _consume_events(agent: Any, event_log: Any, cb: CallbackManager) -> None:
    """Translate the kernel's event stream onto Agent's existing
    CallbackHandler hooks and agent.scratchpad, live, while engine.run()
    is in flight in a sibling task. Tool events accumulate per iteration
    and flush into agent.scratchpad as one "Step N: ..." block, either
    when the next iteration starts or when the run ends -- mirroring
    backend="legacy"'s one-block-per-iteration scratchpad shape closely
    enough for scratchpad-reading middleware, though not byte-identical.
    """
    current_iteration = 0
    pending_tool_events: List[Dict[str, Any]] = []
    tool_inputs: Dict[str, Dict[str, Any]] = {}

    def _flush() -> None:
        nonlocal pending_tool_events
        if pending_tool_events:
            agent.scratchpad += _render_scratchpad_step(current_iteration, pending_tool_events)
            pending_tool_events = []

    async for event in event_log.events():
        if event.type == "iteration_start":
            _flush()
            current_iteration = event.payload.get("iteration", current_iteration) + 1
            cb.fire_iteration_start(current_iteration, agent=agent)
        elif event.type == "tool_start":
            tool_name = event.payload.get("tool", "")
            tool_input = event.payload.get("arguments", {})
            tool_inputs[tool_name] = tool_input
            cb.fire_tool_start(tool_name, tool_input, agent=agent)
        elif event.type == "tool_end":
            tool_name = event.payload.get("tool", "")
            content = event.payload.get("content")
            if content is None:
                content = (
                    ""
                    if event.payload.get("ok", True)
                    else "Tool call was denied by the approval callback."
                )
            cb.fire_tool_end(tool_name, content, agent=agent)
            pending_tool_events.append({
                "tool": tool_name,
                "arguments": tool_inputs.get(tool_name, {}),
                "content": content,
            })
        elif event.type == "llm_end":
            cb.fire_llm_end(event.payload.get("text"), agent=agent, raw=None)
        elif event.type == "empty_response":
            cb.fire_parse_error(current_iteration, "<empty response>", agent=agent)
        elif event.type in ("run_end", "run_cancelled"):
            _flush()
            return


def _build_engine_and_run(agent: Any, query: str, max_iterations: int, kernel: Any) -> Any:
    from autourgos_core import Budget

    memory_context = _get_memory_context(getattr(agent, "memory", None), query)
    system_prompt = getattr(agent, "system_prompt", "") or ""
    if memory_context:
        system_prompt = (system_prompt + "\n\n" + memory_context) if system_prompt else memory_context

    config = kernel.EngineConfig(
        max_iterations=max_iterations,
        max_execution_seconds=getattr(agent, "max_execution_time", None),
        max_consecutive_empty=getattr(agent, "max_consecutive_parse_errors", 3),
        tool_timeout=getattr(agent, "tool_timeout", None),
        max_output_chars=getattr(agent, "MAX_TOOL_OUTPUT_CHARS", 5000),
    )
    context_manager = kernel.CharBudgetContextManager(
        max_chars=getattr(agent, "MAX_SCRATCHPAD_CHARS", 15000),
        max_tokens=getattr(agent, "max_scratchpad_tokens", None),
        token_counter=getattr(agent, "token_counter", None),
    )
    engine = kernel.Engine(
        llm=agent.llm,
        tools=list(getattr(agent, "tools", [])),
        capabilities=list(getattr(agent, "capabilities", [])),
        context_manager=context_manager,
        approval_callback=getattr(agent, "approval_callback", None),
        config=config,
    )
    run = kernel.Run(
        goal=query,
        system_prompt=system_prompt,
        budget=Budget(
            max_iterations=max_iterations,
            max_execution_seconds=getattr(agent, "max_execution_time", None),
            max_effects=getattr(agent, "max_effects", None),
        ),
    )
    policy_executor = None
    policy_factory = getattr(agent, "policy_executor_factory", None)
    if policy_factory is not None:
        policy_executor = policy_factory(run)
        if policy_executor is None or not callable(getattr(policy_executor, "execute", None)):
            raise ValueError("policy_executor_factory(run) must return a policy executor.")
    return engine, run, policy_executor


async def ainvoke_kernel(
    agent: Any,
    query: str,
    max_iterations: int,
    extra_kwargs: Dict[str, Any],
) -> str:
    kernel = _import_kernel()
    engine, run, policy_executor = _build_engine_and_run(agent, query, max_iterations, kernel)

    agent.scratchpad = ""
    event_log = kernel.EventLog()
    cb: CallbackManager = agent.callback_manager

    consumer_task = asyncio.create_task(_consume_events(agent, event_log, cb))
    try:
        result = await engine.run(
            run,
            event_log=event_log,
            policy_executor=policy_executor,
        )
    except Exception as exc:
        raise _translate_error(exc, kernel) from exc
    finally:
        event_log.close()
        await consumer_task

    memory = getattr(agent, "memory", None)
    if memory:
        _record_agent_message(memory, result)

    cb.fire_agent_end(result, agent=agent)
    return result


def invoke_kernel(
    agent: Any,
    query: str,
    max_iterations: int,
    extra_kwargs: Dict[str, Any],
) -> str:
    return asyncio.run(ainvoke_kernel(agent, query, max_iterations, extra_kwargs))
