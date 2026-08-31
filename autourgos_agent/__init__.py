"""
autourgos-agent — Self-contained, general-purpose LLM agent for the Autourgos framework.

Works with any OpenAI-compatible LLM via autourgos-openaichat or
autourgos-responses (or any object with .invoke() / .ainvoke()).

Quick start::

    from autourgos_agent import Agent, tool
    from autourgos_openaichat  import OpenAIChatModel   # or OpenAIResponse

    @tool
    def search(query: str) -> str:
        # "Search the web for information." (docstring, becomes the tool's description)
        return f"Results for: {query}"

    agent = Agent(llm=OpenAIChatModel(model="gpt-4o"), verbose=True)
    agent.add_tools(search)
    result = agent.invoke("What is the latest news about AI?")
    print(result)
"""

from .agent   import Agent
from .base    import (
    BaseLLM,
    BaseAgent,
    AgentLoopMixin,
    CallbackHandler,
    CallbackManager,
    MemoryProtocol,
    AgentError,
    AgentTimeoutError,
    AgentMaxIterationsError,
    AgentParseError,
    AgentLLMError,
    AgentEmptyResponseError,
)
from .logging import AgentLogger
from .runtime import build_tool_list, parse_json_object
from .tool    import tool, Tool

# v1 backward-compat alias
import warnings as _warnings


def Create_Agent(*args: object, **kwargs: object) -> Agent:
    """Deprecated v1 alias. Use Agent instead."""
    _warnings.warn(
        "`Create_Agent` is renamed to `Agent` in v2. "
        "Update your code: `from autourgos_agent import Agent`",
        DeprecationWarning,
        stacklevel=2,
    )
    return Agent(*args, **kwargs)


try:
    from importlib.metadata import version as _meta_version, PackageNotFoundError as _PNF
    __version__ = _meta_version("autourgos-agent")
except Exception:
    __version__ = "2.7.2"

__all__ = [
    "Agent",
    "Create_Agent",
    # base classes
    "BaseLLM",
    "BaseAgent",
    "AgentLoopMixin",
    "CallbackHandler",
    "CallbackManager",
    "MemoryProtocol",
    # exceptions
    "AgentError",
    "AgentTimeoutError",
    "AgentMaxIterationsError",
    "AgentParseError",
    "AgentLLMError",
    "AgentEmptyResponseError",
    # utilities
    "AgentLogger",
    "build_tool_list",
    "parse_json_object",
    # tool decorator
    "tool",
    "Tool",
]
