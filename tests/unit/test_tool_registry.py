import pytest

from app.core.errors import ToolError
from app.tools.base import Tool, ToolResult
from app.tools.registry import ToolRegistry


class DummyTool(Tool):
    name = "dummy.echo"
    description = "Echoes back its input."
    input_schema = {"text": "string"}
    output_schema = {"text": "string"}

    def execute(self, arguments):
        return ToolResult(success=True, data={"text": arguments.get("text", "")})


def test_register_and_get_tool():
    registry = ToolRegistry()
    tool = DummyTool()
    registry.register(tool)

    fetched = registry.get("dummy.echo")
    assert fetched is tool

    result = fetched.execute({"text": "hi"})
    assert result.success is True
    assert result.data == {"text": "hi"}


def test_register_duplicate_tool_raises():
    registry = ToolRegistry()
    registry.register(DummyTool())
    with pytest.raises(ToolError):
        registry.register(DummyTool())


def test_get_unknown_tool_raises():
    registry = ToolRegistry()
    with pytest.raises(ToolError):
        registry.get("does.not.exist")


def test_describe_all():
    registry = ToolRegistry()
    registry.register(DummyTool())
    descriptions = registry.describe_all()
    assert descriptions == [
        {
            "name": "dummy.echo",
            "description": "Echoes back its input.",
            "input_schema": {"text": "string"},
            "output_schema": {"text": "string"},
            "permissions": [],
            "requires_confirmation": False,
        }
    ]
