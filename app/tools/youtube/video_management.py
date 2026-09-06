"""
`youtube.video.*` management tools (master spec Phase 5: video management -
"where officially supported": update metadata, delete, ratings/likes).

Upload is deliberately NOT implemented here. YouTube's resumable upload
protocol needs actual video file bytes streamed through the backend
(`videos.insert` with multipart/resumable upload) - there is no video
capture/file-upload surface anywhere else in Nova yet (voice/text command
input only), so there is nothing for this tool to receive an upload
*from*. Adding it now would mean fabricating an unused code path (master
spec section 1: "implement ONLY the requested phase" / do not build
speculative surface). It's a natural fit once Nova has any file-upload
entry point.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError
from app.integrations.youtube_api import YouTubeApiClient
from app.tools.base import Tool, ToolResult

_YOUTUBE_SCOPE = "https://www.googleapis.com/auth/youtube"
_VALID_RATINGS = {"like", "dislike", "none"}


def _require_access_token(arguments: dict[str, Any]) -> str:
    access_token = arguments.get("access_token")
    if not access_token:
        raise ValidationError("This operation requires a connected Google account (`access_token`).")
    return access_token


class UpdateVideoTool(Tool):
    name = "youtube.video.update"
    description = "Update metadata (title, description, tags) on one of the connected account's own videos."
    input_schema = {
        "video_id": {"type": "string", "required": True},
        "title": {"type": "string", "required": False, "description": "Omit to keep the current title."},
        "description": {"type": "string", "required": False, "description": "Omit to keep the current description."},
        "tags": {"type": "array", "required": False},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"video": {"type": "object"}}
    permissions = [_YOUTUBE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        video_id = (arguments.get("video_id") or "").strip()
        if not video_id:
            raise ValidationError("`video_id` is required for youtube.video.update.")
        access_token = _require_access_token(arguments)

        # `videos.update` replaces the whole snippet - fetch the current
        # one first so an edit that only sets `title` doesn't blank out
        # the description/category (master spec section 13: never silently
        # discard data the caller didn't ask to change).
        current = self._client.get_videos(video_ids=[video_id], access_token=access_token)
        if not current:
            raise ValidationError(f"No video found with id '{video_id}' on the connected account.")
        existing = current[0]

        title = arguments.get("title")
        title = title if title is not None else existing.title
        description = arguments.get("description")
        description = description if description is not None else existing.description
        category_id = existing.category_id or "22"  # YouTube requires a categoryId; 22 = "People & Blogs" fallback

        updated = self._client.update_video(
            video_id=video_id,
            title=title,
            description=description,
            category_id=category_id,
            tags=arguments.get("tags"),
            access_token=access_token,
        )
        return ToolResult(success=True, data={"video": updated.to_dict()})


class DeleteVideoTool(Tool):
    name = "youtube.video.delete"
    description = "Permanently delete one of the connected account's own videos."
    input_schema = {"video_id": {"type": "string", "required": True}, "access_token": {"type": "string", "required": True}}
    output_schema = {"deleted": {"type": "boolean"}}
    permissions = [_YOUTUBE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        video_id = (arguments.get("video_id") or "").strip()
        if not video_id:
            raise ValidationError("`video_id` is required for youtube.video.delete.")
        access_token = _require_access_token(arguments)

        self._client.delete_video(video_id=video_id, access_token=access_token)
        return ToolResult(success=True, data={"deleted": True, "video_id": video_id})


class RateVideoTool(Tool):
    name = "youtube.video.rate"
    description = "Like, dislike, or clear your rating on a video."
    input_schema = {
        "video_id": {"type": "string", "required": True},
        "rating": {"type": "string", "required": True, "enum": sorted(_VALID_RATINGS)},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"rating": {"type": "string"}}
    permissions = [_YOUTUBE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        video_id = (arguments.get("video_id") or "").strip()
        rating = arguments.get("rating")
        if not video_id or rating not in _VALID_RATINGS:
            raise ValidationError(f"`video_id` and `rating` (one of {sorted(_VALID_RATINGS)}) are required for youtube.video.rate.")
        access_token = _require_access_token(arguments)

        self._client.rate_video(video_id=video_id, rating=rating, access_token=access_token)
        return ToolResult(success=True, data={"video_id": video_id, "rating": rating})


class GetVideoRatingTool(Tool):
    name = "youtube.video.get_rating"
    description = "Get the connected account's current rating (like/dislike/none) on a video."
    input_schema = {"video_id": {"type": "string", "required": True}, "access_token": {"type": "string", "required": True}}
    output_schema = {"rating": {"type": "string", "nullable": True}}
    permissions = [_YOUTUBE_SCOPE]
    requires_confirmation = False

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        video_id = (arguments.get("video_id") or "").strip()
        if not video_id:
            raise ValidationError("`video_id` is required for youtube.video.get_rating.")
        access_token = _require_access_token(arguments)

        rating = self._client.get_video_rating(video_id=video_id, access_token=access_token)
        return ToolResult(success=True, data={"video_id": video_id, "rating": rating})
