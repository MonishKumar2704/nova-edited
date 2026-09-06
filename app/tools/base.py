"""
Tool interface.

Concrete tools (YouTube search, Gmail send, ...) implement this ABC and
register themselves with a `ToolRegistry`. The agent orchestrator depends
only on this interface, never on a concrete tool implementation, so that
new tools can be added without modifying the orchestrator (master spec
section 7 - DYNAMIC TOOL SYSTEM).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Structured result returned by every tool execution."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_code: str | None = None


class Tool(ABC):
    """Base class every Nova tool must implement."""

    #: unique, dotted tool name, e.g. "youtube.search" or "gmail.send"
    name: str = ""
    #: human/LLM-readable description of what the tool does
    description: str = ""
    #: JSON-schema-like dict describing accepted arguments
    input_schema: dict[str, Any] = {}
    #: JSON-schema-like dict describing the shape of ToolResult.data
    output_schema: dict[str, Any] = {}
    #: OAuth scopes or internal permissions required to run this tool
    permissions: list[str] = []
    #: if True, the orchestrator must obtain explicit user confirmation
    #: before executing this tool (see master spec section 39)
    requires_confirmation: bool = False

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate `arguments` against `input_schema` and perform the action.

        Implementations must never raise raw exceptions for expected
        failure modes (auth, network, validation, etc). Instead they
        should catch them and return a `ToolResult(success=False, ...)`,
        or raise one of the classified errors in `app.core.errors` so the
        API layer can translate it consistently.
        """
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Structured definition suitable for handing to an LLM as a tool spec."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "permissions": self.permissions,
            "requires_confirmation": self.requires_confirmation,
        }
