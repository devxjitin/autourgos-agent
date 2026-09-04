"""
logging.py -- AgentLogger for autourgos-agent.

Provides a LangChain-style verbose console trace: a coloured
"> Starting X..." / "> X finished." banner wrapping the run,
with each step printed as Thought / Action / Action Input / Observation --
the same vocabulary the agent prompt itself uses. Suppressed entirely when
verbose=False so production code stays silent.
"""

from __future__ import annotations

from typing import Any


# ANSI colour codes (fall back gracefully if the terminal doesn't support them)
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RED    = "\033[91m"
_GREY    = "\033[90m"
_BLUE    = "\033[94m"
_MAGENTA = "\033[95m"


class AgentLogger:
    """
    Structured, LangChain-style logger for Agent agent execution.

    Output looks like::

        > Starting Agent...

        Thought: I need to add 123 and 456. I'll use the calculator tool.
        Action: calculator
        Action Input: {'a': 123, 'b': 456}
        Observation: 579
        Thought: I have the result from the calculator.
        Final Answer: 123 + 456 = 579

        > Agent finished.

    Middleware (toolbox, summarizer, hcix, preiteration, ...) can also
    narrate what it does via ``agent.logger.middleware(source, message)``,
    printed in magenta with a ``[Source]`` prefix so it's unambiguous which
    middleware produced the line, e.g.::

        [Toolbox] Exposed toolbox 'search_tools' to agent.
        [Summarizer] Compressed scratchpad (iteration 5, was 15,320 chars).

    Parameters
    ----------
    verbose : bool
        Print output to stdout when True. Silent when False.
    agent_name : str
        Label shown in the run-start / run-end banner.
    full_output : bool
        When True, also print the raw LLM response at each step.
        Useful for debugging prompt/parse issues.
    """

    def __init__(
        self,
        verbose: bool = False,
        agent_name: str = "Agent",
        full_output: bool = False,
    ) -> None:
        self.verbose      = verbose
        self.agent_name   = agent_name
        self.full_output  = full_output

    # -- internal ----------------------------------------------------------------

    def _emit(self, colour: str, text: str) -> None:
        if self.verbose:
            print(f"{colour}{text}{_RESET}")

    def _truncate(self, text: str, limit: int = 300) -> str:
        text = "" if text is None else str(text)
        return text if len(text) <= limit else text[:limit] + f"{_DIM}... [truncated]{_RESET}"

    # -- run lifecycle --------------------------------------------------------------

    def run_start(self, query: str) -> None:
        """Printed once, right when a run begins. Not literally a LangChain
        "chain" -- Agent runs a agent loop, not a chain -- so the
        banner says "Starting <agent_name>..." rather than borrowing
        LangChain's chain terminology, while keeping the same
        Thought/Action/Observation trace style below it."""
        if self.verbose:
            print()
        self._emit(f"{_GREEN}{_BOLD}", f"> Starting {self.agent_name}...")
        if self.verbose:
            print()

    def run_end(self) -> None:
        """Printed once, right when a run ends (success, error, timeout, or
        max-iterations)."""
        if self.verbose:
            print()
        self._emit(f"{_GREEN}{_BOLD}", f"> {self.agent_name} finished.")

    # -- public log methods --------------------------------------------------------

    def thought(self, thought: str, iteration: int) -> None:
        self._emit(_CYAN, f"Thought: {thought}")

    def tool_call(self, tool_name: str, tool_input: Any, iteration: int) -> None:
        self._emit(f"{_YELLOW}{_BOLD}", f"Action: {tool_name}")
        self._emit(_YELLOW, f"Action Input: {tool_input}")

    def tool_result(self, tool_name: str, result: str, iteration: int) -> None:
        self._emit(_BLUE, f"Observation: {self._truncate(result)}")

    def final_answer(self, answer: str) -> None:
        self._emit(f"{_GREEN}{_BOLD}", f"Final Answer: {answer}")

    def parse_error(self, raw_response: str, iteration: int) -> None:
        self._emit(_RED, f"Parse Error: {self._truncate(raw_response, 200)}")

    def llm_response(self, response: str, iteration: int) -> None:
        """Only printed when full_output=True."""
        if self.full_output:
            self._emit(_GREY, f"LLM Raw: {self._truncate(response, 500)}")

    def memory_action(self, message: str) -> None:
        self._emit(_GREY, f"Memory: {message}")

    def info(self, message: str) -> None:
        self._emit(_GREY, f"Info: {message}")

    def warning(self, message: str) -> None:
        """
        Duck-typed to match the standard-library logging.Logger interface's
        .warning(message) -- middleware packages (e.g. autourgos-hcix's
        CognitiveInterruptManager.poll()) pass agent.logger into code that
        expects a logger-shaped object exposing at least .info()/.warning(),
        matching this class's existing .info() method.
        """
        self._emit(_RED, f"Warning: {message}")

    def middleware(self, source: str, message: str) -> None:
        """
        Narrate what a middleware/callback handler is doing, e.g. toolbox
        exposing a tool, summarizer compressing the scratchpad, or hcix
        injecting a human override.

        Middleware packages call this defensively -- ``agent`` is always
        passed to every CallbackHandler hook, so middleware can do
        ``logger = getattr(agent, "logger", None)`` and, if present, call
        ``logger.middleware("Toolbox", "...")`` -- with no hard dependency
        on this class and no crash if ``agent`` isn't a Agent (or has
        no ``.logger``) or verbose is off.

        Parameters
        ----------
        source : str
            Short label identifying which middleware produced the message,
            e.g. "Toolbox", "Summarizer", "HCIx", "PreIteration".
        message : str
            One-line, human-readable description of what happened.
        """
        self._emit(_MAGENTA, f"[{source}] {message}")
