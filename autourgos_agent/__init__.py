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

Lazy-loaded toolboxes::

    from autourgos_core import tool, Toolbox

    @tool
    def web_search(query: str) -> str:
        ...

    web = Toolbox(name="web", description="Web search and page scraping tools.",
                   tools=[web_search, scrape_url])

    agent = Agent(llm=OpenAIChatModel(model="gpt-4o"), toolbox=[web])
"""

from autourgos_core import Toolbox

from .agent   import Agent
from ._toolbox import ToolboxMiddleware
from ._preiteration import PARALLEL, SEQUENTIAL, PreIterationMiddleware, is_async_callable
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
    AgentAlreadyRunningError,
)
from .logging import AgentLogger
from .runtime import build_tool_list, parse_json_object, inject_prompt_block, remove_prompt_block
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


from autourgos_core import package_version

__version__ = package_version("autourgos-agent", fallback="3.9.0")

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
    "AgentAlreadyRunningError",
    # utilities
    "AgentLogger",
    "build_tool_list",
    "parse_json_object",
    "inject_prompt_block",
    "remove_prompt_block",
    # tool decorator
    "tool",
    "Tool",
    # native toolbox support (Agent(toolbox=[...]))
    "Toolbox",
    "ToolboxMiddleware",
    # pre-iteration middleware
    "PreIterationMiddleware",
    "SEQUENTIAL",
    "PARALLEL",
    "is_async_callable",
]
