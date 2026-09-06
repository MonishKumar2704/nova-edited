"""`youtube.channel.my_uploads` tool (master spec Phase 5: own uploaded videos)."""

from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError
from app.integrations.youtube_api import YouTubeApiClient
from app.tools.base import Tool, ToolResult


class MyUploadsTool(Tool):
    name = "youtube.channel.my_uploads"
    description = "List videos uploaded by the connected Google account's own YouTube channel."
    input_schema = {
        "access_token": {"type": "string", "required": True},
        "max_results": {"type": "integer", "required": False, "default": 25},
        "page_token": {"type": "string", "required": False},
    }
    output_schema = {"uploads": {"type": "array", "items": "PlaylistItem"}, "next_page_token": {"type": "string", "nullable": True}}
    permissions = ["https://www.googleapis.com/auth/youtube.readonly"]
    requires_confirmation = False

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        access_token = arguments.get("access_token")
        if not access_token:
            raise ValidationError("`access_token` (a connected Google account) is required for youtube.channel.my_uploads.")

        uploads, next_page_token = self._client.get_my_uploads(
            access_token=access_token,
            max_results=int(arguments.get("max_results", 25)),
            page_token=arguments.get("page_token"),
        )
        return ToolResult(success=True, data={"uploads": [u.to_dict() for u in uploads], "next_page_token": next_page_token})
