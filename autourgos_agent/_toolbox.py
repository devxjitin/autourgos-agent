"""
_toolbox.py -- native Agent(toolbox=[...]) support.

Ported from the (now retired) standalone autourgos-toolbox package: same
lazy-loading behavior (toolbox names/descriptions shown upfront, tools
loaded on demand via expose_toolbox/expose_tool meta-tools), folded
directly into the agent loop as a plain constructor kwarg (Agent(toolbox=
[...]) builds a _ToolboxRuntime internally -- see agent.py) rather than
through the CallbackHandler/middleware bus -- native features of this
package are kwargs, not middleware; that bus is reserved for third-party
extensions (autourgos-hcix, autourgos-skills, your own code). Toolbox
itself lives in autourgos-core (autourgos_core.toolbox) alongside the
@tool decorator.
"""
from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from autourgos_core import Toolbox, infer_json_schema

from .base import _tool_name
from .runtime import inject_prompt_block, remove_prompt_block

__all__ = ["StructuredTool"]

logger = logging.getLogger(__name__)

RESERVED_TOOL_NAMES = {"expose_toolbox", "expose_tool"}


# ── StructuredTool ─────────────────────────────────────────────────────────

class StructuredTool:
    """
    A callable tool with a name, description, and auto-inferred JSON schema
    built from the function's type annotations and docstring.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        args_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name        = name
        self.description = description
        self.func        = func
        self.args_schema = args_schema or self._infer_schema(func)

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "StructuredTool":
        tool_name = name or func.__name__
        tool_desc = description or (inspect.getdoc(func) or "")
        return cls(name=tool_name, description=tool_desc, func=func)

    @staticmethod
    def _infer_schema(func: Callable) -> Dict[str, Any]:
        # Delegates to autourgos_core.infer_json_schema -- the canonical
        # inference implementation shared across the framework (handles
        # Optional/Union unwrapping and stringified annotations, which this
        # package's own prior standalone implementation did not).
        return infer_json_schema(func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def _register_tool(registry: Dict[str, Dict[str, Any]], tool: Any) -> None:
    """Register a tool (StructuredTool or plain callable) into a dict registry."""
    if isinstance(tool, StructuredTool):
        registry[tool.name] = {
            "description": tool.description,
            "parameters":  tool.args_schema,
            "func":        tool.func,
        }
    elif callable(tool):
        name = getattr(tool, "__name__", str(tool))
        doc  = inspect.getdoc(tool) or ""
        registry[name] = {
            "description": doc,
            "parameters":  StructuredTool._infer_schema(tool),
            "func":        tool,
        }
    elif isinstance(tool, dict) and "name" in tool:
        registry[tool["name"]] = tool


def _tool_registry_list(tools: Dict[str, Dict[str, Any]]) -> str:
    """Format a tool registry into a prompt-ready string."""
    lines: List[str] = []
    for name, info in tools.items():
        desc   = info.get("description", "")
        params = info.get("parameters", {})
        lines.append(f"Tool: {name}\nDescription: {desc}\nParameters: {json.dumps(params, indent=2)}\n")
    return "\n".join(lines)


def _to_agent_tool_dict(tool: Any) -> Any:
    """Convert a StructuredTool / plain callable / already-shaped dict into
    the real Agent tool dict shape ({"name", "description", "parameters", "func"})."""
    if isinstance(tool, dict) and "name" in tool:
        return tool
    if isinstance(tool, StructuredTool):
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.args_schema,
            "func": tool.func,
        }
    if callable(tool):
        registry: Dict[str, Dict[str, Any]] = {}
        _register_tool(registry, tool)
        (name, info), = registry.items()
        return {"name": name, "description": info["description"],
                "parameters": info["parameters"], "func": info["func"]}
    return tool


class _ToolboxRuntime:
    """
    Inline (non-middleware) dynamic lazy-loading and sandboxing of
    toolboxes, built directly by Agent(toolbox=[...]) -- mirrors
    _PreIterationRuntime/_HistoryRecorder's pattern: this only ever
    applies to the ONE Agent instance it's configured on, so a flat
    ``self._run_state`` dict is safe (no PerAgentRegistry needed) since
    Agent's own _run_lock already guarantees only one run is ever active
    per instance at a time.
    """

    def __init__(
        self,
        toolboxes: Optional[Union[List[Toolbox], Dict[str, Dict[str, Any]]]] = None,
    ) -> None:
        self.logger    = logging.getLogger(__name__)
        self.toolboxes: Dict[str, Toolbox] = {}

        self._run_state: Optional[Dict[str, Any]] = None

        if toolboxes:
            if isinstance(toolboxes, list):
                for tb in toolboxes:
                    if isinstance(tb, Toolbox):
                        self.add_toolbox(tb.name, tb.description, tb.tools)
            elif isinstance(toolboxes, dict):
                for name, info in toolboxes.items():
                    self.add_toolbox(name, info.get("description", ""), info.get("tools", []))

    # ── public API ────────────────────────────────────────────────────────

    def add_toolbox(self, name: str, description: str, tools: List[Any]) -> None:
        """Register a new toolbox. Can be called before or after agent start."""
        name = name.strip()
        incoming_names = {n for n in (self._get_tool_name(t) for t in tools) if n}
        for other_name, other_tb in self.toolboxes.items():
            if other_name == name:
                continue
            existing_names = {n for n in (self._get_tool_name(t) for t in other_tb.tools) if n}
            collisions = incoming_names & existing_names
            if collisions:
                raise ValueError(
                    f"Toolbox '{name}' has tool name(s) {sorted(collisions)} that "
                    f"already exist in toolbox '{other_name}'. Tool names must be "
                    f"unique across all registered toolboxes."
                )
        self.toolboxes[name] = Toolbox(name, description, tools)

    # ── lifecycle (called directly by Agent.invoke()/ainvoke(), not middleware) ──

    def start(self, agent: Any) -> None:
        """Called once at the start of a run: registers the expose_toolbox/
        expose_tool meta-tools and injects the toolbox catalog into the
        prompt. Raises ValueError if the agent already has a tool named
        expose_toolbox/expose_tool (this feature reserves those names)."""
        existing_names = {self._get_tool_name(t) for t in getattr(agent, "tools", [])}
        conflicts = RESERVED_TOOL_NAMES & existing_names
        if conflicts:
            raise ValueError(
                f"Agent(toolbox=...) cannot start: the agent already has tool(s) named "
                f"{sorted(conflicts)!r}, which this feature reserves for its own "
                f"meta-tools. Rename your tool(s), or remove toolbox=."
            )

        run_state: Dict[str, Any] = {
            "added_tools":     [],
            "injected_blocks": [],
            "exposed":         set(),
            "exposed_tools":   set(),
            "displaced":       {},
        }
        self._run_state = run_state

        def expose_toolbox(toolbox_name: str) -> str:
            """Expose all tools in the specified toolbox, making them available for you to use.

            Args:
                toolbox_name: The exact name of the toolbox to expose (e.g. 'github').
            """
            return self._expose_toolbox_action(toolbox_name, agent=agent)

        def expose_tool(tool_name: str) -> str:
            """Expose a single tool by name, searching across all registered toolboxes.

            Args:
                tool_name: The exact name of the tool to expose (e.g. 'create_pr'). This
                    does not require knowing which toolbox the tool lives in.
            """
            return self._expose_tool_action(tool_name, agent=agent)

        expose_toolbox_tool = StructuredTool.from_function(
            func=expose_toolbox,
            name="expose_toolbox",
            description="Expose all tools in the specified toolbox, making them available for you to use.",
        )
        expose_tool_tool = StructuredTool.from_function(
            func=expose_tool,
            name="expose_tool",
            description=(
                "Expose a single tool by name (searched across all registered toolboxes), "
                "making just that tool available for you to use without loading the rest "
                "of its toolbox."
            ),
        )
        added = [_to_agent_tool_dict(expose_toolbox_tool), _to_agent_tool_dict(expose_tool_tool)]
        agent.add_tools(*added)
        run_state["added_tools"].extend(added)

        if self.toolboxes:
            catalog = "\n".join(
                f"- **{name}**: {tb.description}"
                for name, tb in self.toolboxes.items()
            )
            instruction = (
                "## Dynamic Toolboxes\n"
                "You have access to specialized toolboxes that are NOT loaded by default to keep "
                "the context window clean. If you need tools from any toolbox below, you MUST call "
                "`expose_toolbox(toolbox_name)` or `expose_tool(toolbox_name)` with the exact name "
                "of the toolbox first. Once called, all tools inside that toolbox will be loaded "
                "and available.\n\n"
                f"Available Toolboxes:\n{catalog}\n\n"
                "Do NOT attempt to use any toolbox tools until you have called `expose_toolbox` "
                "and received confirmation."
            )
            run_state["injected_blocks"].append(inject_prompt_block(agent, instruction))

    def restore(self, agent: Any) -> None:
        """Called at the end of a run (success or error): removes the
        meta-tools/injected schemas this run added and restores any tool
        it temporarily displaced, so the agent returns to its exact
        pre-run state before the next invoke()/ainvoke()."""
        self._restore_agent(agent)

    # ── internal ──────────────────────────────────────────────────────────

    def _expose_toolbox_action(self, toolbox_name: str, agent: Any) -> str:
        run_state = self._run_state
        if agent is None or run_state is None:
            return "Error: No active agent reference found."

        toolbox_name = toolbox_name.strip()
        if toolbox_name not in self.toolboxes:
            available = ", ".join(self.toolboxes.keys())
            return f"Error: Toolbox '{toolbox_name}' not found. Available: {available}"

        if toolbox_name in run_state["exposed"]:
            return f"Toolbox '{toolbox_name}' is already exposed and its tools are available."

        tb = self.toolboxes[toolbox_name]
        added = [_to_agent_tool_dict(t) for t in tb.tools]
        self._capture_displaced(agent, run_state, added)
        agent.add_tools(*added)
        run_state["added_tools"].extend(added)

        new_registry: Dict[str, Dict[str, Any]] = {}
        for tool in tb.tools:
            _register_tool(new_registry, tool)
        schemas = _tool_registry_list(new_registry)

        exposure_note = (
            f"\n\n### Exposed Toolbox: '{toolbox_name}'\n"
            f"The following tools are now active and ready to use:\n{schemas}\n"
        )
        run_state["injected_blocks"].append(inject_prompt_block(agent, exposure_note))

        run_state["exposed"].add(toolbox_name)
        self.logger.info(f"Exposed toolbox '{toolbox_name}' to agent.")
        logger_ = getattr(agent, "logger", None)
        if logger_:
            logger_.middleware("Toolbox", f"Exposed toolbox '{toolbox_name}' to agent.")
        return f"Success: Exposed all tools in '{toolbox_name}' toolbox. You can now call them."

    def _capture_displaced(self, agent: Any, run_state: Dict[str, Any], incoming: List[Any]) -> None:
        existing_by_name = {self._get_tool_name(t): t for t in getattr(agent, "tools", [])}
        for tool in incoming:
            name = self._get_tool_name(tool)
            if name is None or name in run_state["displaced"]:
                continue
            existing = existing_by_name.get(name)
            if existing is not None:
                run_state["displaced"][name] = existing

    @staticmethod
    def _get_tool_name(tool: Any) -> Optional[str]:
        return _tool_name(tool)

    def _find_tool(self, tool_name: str) -> Optional[Any]:
        for tb in self.toolboxes.values():
            for tool in tb.tools:
                if self._get_tool_name(tool) == tool_name:
                    return tool
        return None

    def _expose_tool_action(self, tool_name: str, agent: Any) -> str:
        run_state = self._run_state
        if agent is None or run_state is None:
            return "Error: No active agent reference found."

        tool_name = tool_name.strip()

        if tool_name in run_state["exposed_tools"]:
            return f"Tool '{tool_name}' is already exposed and available."

        tool = self._find_tool(tool_name)
        if tool is None:
            return f"Error: Tool '{tool_name}' not found in any registered toolbox."

        converted = _to_agent_tool_dict(tool)
        self._capture_displaced(agent, run_state, [converted])
        agent.add_tools(converted)
        run_state["added_tools"].append(converted)

        new_registry: Dict[str, Dict[str, Any]] = {}
        _register_tool(new_registry, tool)
        schemas = _tool_registry_list(new_registry)

        exposure_note = (
            f"\n\n### Exposed Tool: '{tool_name}'\n"
            f"The following tool is now active and ready to use:\n{schemas}\n"
        )
        run_state["injected_blocks"].append(inject_prompt_block(agent, exposure_note))

        run_state["exposed_tools"].add(tool_name)
        self.logger.info(f"Exposed tool '{tool_name}' to agent.")
        logger_ = getattr(agent, "logger", None)
        if logger_:
            logger_.middleware("Toolbox", f"Exposed tool '{tool_name}' to agent.")
        return f"Success: Exposed tool '{tool_name}'. You can now call it."

    def _restore_agent(self, agent: Any = None) -> None:
        if agent is None:
            return
        run_state = self._run_state
        self._run_state = None
        if run_state is None:
            return
        if hasattr(agent, "tools") and run_state["added_tools"]:
            added_ids = {id(t) for t in run_state["added_tools"]}
            agent.tools = [t for t in agent.tools if id(t) not in added_ids]
        if hasattr(agent, "tools") and run_state["displaced"]:
            current_names = {self._get_tool_name(t) for t in agent.tools}
            for name, original in run_state["displaced"].items():
                if name not in current_names:
                    agent.tools.append(original)
        for block in run_state["injected_blocks"]:
            remove_prompt_block(agent, block)
