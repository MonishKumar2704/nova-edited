"""
Tool registry: dynamic discovery of available tools.

Future tool packages register their tools here (see `app.tools.youtube`
and `app.tools.gmail`, currently empty placeholders). The agent
orchestrator queries the registry rather than importing concrete tools
directly, so adding a new tool module does not require modifying the
orchestrator (master spec section 7).
"""

from __future__ import annotations

from app.core.errors import ToolError
from app.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ToolError(f"Tool {tool.__class__.__name__} must define a non-empty `name`.")
        if tool.name in self._tools:
            raise ToolError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"Unknown tool '{name}'.") from exc

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def describe_all(self) -> list[dict]:
        return [tool.describe() for tool in self._tools.values()]


_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Process-wide singleton registry.

    Concrete tools are added to this registry in later phases
    (Phase 3+ for YouTube, Phase 6+ for Gmail). No tools are registered
    yet in Phase 0.
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
