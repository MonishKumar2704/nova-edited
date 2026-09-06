"""
`youtube.list_channels` tool: the connected Google account's own channel.

Unlike search/get_video, this always requires an OAuth access token
(`mine=true` has no API-key equivalent) - callers (API layer / future
agent orchestrator) are responsible for resolving a valid access token via
`GoogleAuthService.get_valid_access_token()` and passing it in as
`access_token`. The tool itself has no Flask/session knowledge, matching
every other tool in this package.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import AuthenticationError
from app.integrations.youtube_api import YouTubeApiClient
from app.tools.base import Tool, ToolResult


class ListMyChannelTool(Tool):
    name = "youtube.list_channels"
    description = "Get the connected Google account's own YouTube channel info (title, description, subscriber/video counts)."
    input_schema = {"access_token": {"type": "string", "required": True}}
    output_schema = {"channel": {"type": "object", "nullable": True}}
    permissions = ["https://www.googleapis.com/auth/youtube.readonly"]
    requires_confirmation = False

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        access_token = arguments.get("access_token")
        if not access_token:
            raise AuthenticationError("youtube.list_channels requires a connected Google account.")

        channel = self._client.get_my_channel(access_token=access_token)
        return ToolResult(success=True, data={"channel": channel.to_dict() if channel else None})
