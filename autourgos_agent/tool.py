"""
tool.py -- @tool decorator, re-exported from autourgos-core.

The decorator's definition moved to autourgos-core (autourgos_core.tool) so
it can be shared with autourgos-core's own Toolbox without a circular
dependency between autourgos-core and autourgos-agent. This module keeps
`from autourgos_agent import tool` / `from autourgos_agent.tool import tool`
working unchanged for existing code -- `from autourgos_core import tool` is
the new canonical import path.
"""

from __future__ import annotations

from autourgos_core import Tool, tool

__all__ = ["tool", "Tool"]
