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

from typing import Any, Callable, Dict, List, Optional

from .base    import AgentLoopMixin, BaseAgent, BaseLLM, CallbackHandler, MemoryProtocol
from .logging import AgentLogger
from .prompt  import PREFIX_PROMPT, LOGIC_PROMPT, SUFFIX_PROMPT
from .runtime import build_tool_list, parse_json_object


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
    approval_callback : callable, optional
        Called before each tool execution as approval_callback(tool_name, tool_input).
        Return a truthy value to allow, falsy to deny.
    middleware : list[CallbackHandler], optional
        Event hooks for agent lifecycle events.
    max_consecutive_parse_errors : int
        Number of consecutive JSON parse failures before giving up.
    tools : list[dict], optional
        Initial tool list. More can be added with add_tools().
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
        approval_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        middleware: Optional[List[CallbackHandler]] = None,
        max_consecutive_parse_errors: int = 3,
        tools: Optional[List[Any]] = None,
        system_prompt: str = "",
        tool_calling_mode: str = "prompt",
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
        self.approval_callback = approval_callback
        self.max_consecutive_parse_errors = max_consecutive_parse_errors
        self.system_prompt   = system_prompt
        self.prompt_template = PREFIX_PROMPT + LOGIC_PROMPT + SUFFIX_PROMPT
        self.logger = AgentLogger(
            verbose=verbose,
            agent_name="Agent",
            full_output=full_output,
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
        if not self.tools:
            raise ValueError("No tools added. Call agent.add_tools(tool) first.")

        self.current_query = query

        try:
            if self.memory:
                self.memory.add_user_message(query)
                self.logger.memory_action("Added user message to memory.")

            self.callback_manager.fire_agent_start(query, agent=self)
            self.logger.run_start(query)

            if self.tool_calling_mode == "native":
                return self._run_loop_native(
                    query,
                    max_iterations=max_iterations or self.max_iterations,
                    approval_callback=self.approval_callback,
                    extra_kwargs=kwargs,
                )
            return self._run_loop(
                query,
                build_tool_list(self.tools),
                max_iterations=max_iterations or self.max_iterations,
                approval_callback=self.approval_callback,
                extra_kwargs=kwargs,
            )
        except Exception as exc:
            self.callback_manager.fire_agent_error(exc, agent=self)
            raise
        finally:
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
        if not self.tools:
            raise ValueError("No tools added. Call agent.add_tools(tool) first.")

        self.current_query = query

        try:
            if self.memory:
                self.memory.add_user_message(query)
                self.logger.memory_action("Added user message to memory.")

            self.callback_manager.fire_agent_start(query, agent=self)
            self.logger.run_start(query)

            if self.tool_calling_mode == "native":
                return await self._arun_loop_native(
                    query,
                    max_iterations=max_iterations or self.max_iterations,
                    approval_callback=self.approval_callback,
                    extra_kwargs=kwargs,
                )
            return await self._arun_loop(
                query,
                build_tool_list(self.tools),
                max_iterations=max_iterations or self.max_iterations,
                approval_callback=self.approval_callback,
                extra_kwargs=kwargs,
            )
        except Exception as exc:
            self.callback_manager.fire_agent_error(exc, agent=self)
            raise
        finally:
            self.logger.run_end()
