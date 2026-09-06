"""`youtube.get_video` tool: fetch metadata for one or more known video IDs."""

from __future__ import annotations

from typing import Any

from app.core.cache import TTLCache, make_cache_key
from app.core.errors import ValidationError
from app.integrations.youtube_api import YouTubeApiClient
from app.tools.base import Tool, ToolResult


class GetVideoTool(Tool):
    name = "youtube.get_video"
    description = "Fetch metadata (title, channel, duration, view/like counts) for one or more known YouTube video IDs."
    input_schema = {
        "video_id": {"type": "string", "required": False, "description": "A single video ID."},
        "video_ids": {"type": "array", "required": False, "description": "Multiple video IDs (max 50)."},
    }
    output_schema = {"results": {"type": "array", "items": "VideoDetails"}}
    permissions = []
    requires_confirmation = False

    def __init__(self, *, client: YouTubeApiClient, cache: TTLCache | None = None) -> None:
        self._client = client
        self._cache = cache

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        video_ids = list(arguments.get("video_ids") or [])
        single = arguments.get("video_id")
        if single:
            video_ids.append(single)
        video_ids = [v for v in dict.fromkeys(video_ids) if v]  # de-dupe, preserve order

        if not video_ids:
            raise ValidationError("`video_id` or `video_ids` is required for youtube.get_video.")

        access_token = arguments.get("access_token")
        cache_key = make_cache_key("youtube.get_video", ",".join(video_ids), bool(access_token))

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return ToolResult(success=True, data={"results": cached, "cached": True})

        videos = self._client.get_videos(video_ids=video_ids, access_token=access_token)
        results = [v.to_dict() for v in videos]

        if self._cache is not None:
            self._cache.set(cache_key, results)

        return ToolResult(success=bool(results), data={"results": results, "cached": False})
