"""
`youtube.playlist.*` tools (master spec Phase 5: playlists - list,
retrieve, create, update, add/remove videos, reorder).

Every playlist mutation requires a connected Google account (there is no
API-key path for writes - see `YouTubeApiClient._write`) and is marked
`requires_confirmation = True`: these have real, visible side effects on
the user's actual YouTube account (master spec section 39 - confirmation
architecture applies to any tool with meaningful external side effects,
not just Gmail sends). Read-only listing/retrieval is not.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import ValidationError
from app.integrations.youtube_api import YouTubeApiClient
from app.tools.base import Tool, ToolResult

_YOUTUBE_SCOPE = "https://www.googleapis.com/auth/youtube"
_VALID_PRIVACY = {"private", "public", "unlisted"}


def _require_access_token(arguments: dict[str, Any]) -> str:
    access_token = arguments.get("access_token")
    if not access_token:
        raise ValidationError("This operation requires a connected Google account (`access_token`).")
    return access_token


class ListMyPlaylistsTool(Tool):
    name = "youtube.playlist.list"
    description = "List the connected Google account's own YouTube playlists."
    input_schema = {
        "access_token": {"type": "string", "required": True},
        "max_results": {"type": "integer", "required": False, "default": 25},
        "page_token": {"type": "string", "required": False},
    }
    output_schema = {"playlists": {"type": "array", "items": "PlaylistSummary"}, "next_page_token": {"type": "string", "nullable": True}}
    permissions = [_YOUTUBE_SCOPE]
    requires_confirmation = False

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        access_token = _require_access_token(arguments)
        max_results = int(arguments.get("max_results", 25))
        playlists, next_page_token = self._client.list_my_playlists(
            access_token=access_token, max_results=max_results, page_token=arguments.get("page_token")
        )
        return ToolResult(
            success=True,
            data={"playlists": [p.to_dict() for p in playlists], "next_page_token": next_page_token},
        )


class GetPlaylistTool(Tool):
    name = "youtube.playlist.get"
    description = "Retrieve one playlist's details and its videos."
    input_schema = {
        "playlist_id": {"type": "string", "required": True},
        "access_token": {"type": "string", "required": False},
        "max_results": {"type": "integer", "required": False, "default": 25},
        "page_token": {"type": "string", "required": False},
    }
    output_schema = {
        "playlist": {"type": "object", "nullable": True},
        "items": {"type": "array", "items": "PlaylistItem"},
        "next_page_token": {"type": "string", "nullable": True},
    }
    permissions = []
    requires_confirmation = False

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        playlist_id = (arguments.get("playlist_id") or "").strip()
        if not playlist_id:
            raise ValidationError("`playlist_id` is required for youtube.playlist.get.")

        access_token = arguments.get("access_token")
        playlist = self._client.get_playlist(playlist_id=playlist_id, access_token=access_token)
        items, next_page_token = self._client.list_playlist_items(
            playlist_id=playlist_id,
            max_results=int(arguments.get("max_results", 25)),
            page_token=arguments.get("page_token"),
            access_token=access_token,
        )
        return ToolResult(
            success=True,
            data={
                "playlist": playlist.to_dict() if playlist else None,
                "items": [i.to_dict() for i in items],
                "next_page_token": next_page_token,
            },
        )


class CreatePlaylistTool(Tool):
    name = "youtube.playlist.create"
    description = "Create a new YouTube playlist on the connected account."
    input_schema = {
        "title": {"type": "string", "required": True},
        "description": {"type": "string", "required": False, "default": ""},
        "privacy_status": {"type": "string", "required": False, "default": "private", "enum": sorted(_VALID_PRIVACY)},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"playlist": {"type": "object"}}
    permissions = [_YOUTUBE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        title = (arguments.get("title") or "").strip()
        if not title:
            raise ValidationError("`title` is required for youtube.playlist.create.")
        privacy_status = arguments.get("privacy_status", "private")
        if privacy_status not in _VALID_PRIVACY:
            raise ValidationError(f"`privacy_status` must be one of {sorted(_VALID_PRIVACY)}.")
        access_token = _require_access_token(arguments)

        playlist = self._client.create_playlist(
            title=title,
            description=arguments.get("description", ""),
            privacy_status=privacy_status,
            access_token=access_token,
        )
        return ToolResult(success=True, data={"playlist": playlist.to_dict()})


class UpdatePlaylistTool(Tool):
    name = "youtube.playlist.update"
    description = "Update an existing playlist's title, description, or privacy."
    input_schema = {
        "playlist_id": {"type": "string", "required": True},
        "title": {"type": "string", "required": True},
        "description": {"type": "string", "required": False, "default": ""},
        "privacy_status": {"type": "string", "required": False, "default": "private", "enum": sorted(_VALID_PRIVACY)},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"playlist": {"type": "object"}}
    permissions = [_YOUTUBE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        playlist_id = (arguments.get("playlist_id") or "").strip()
        title = (arguments.get("title") or "").strip()
        if not playlist_id or not title:
            raise ValidationError("`playlist_id` and `title` are required for youtube.playlist.update.")
        privacy_status = arguments.get("privacy_status", "private")
        if privacy_status not in _VALID_PRIVACY:
            raise ValidationError(f"`privacy_status` must be one of {sorted(_VALID_PRIVACY)}.")
        access_token = _require_access_token(arguments)

        playlist = self._client.update_playlist(
            playlist_id=playlist_id,
            title=title,
            description=arguments.get("description", ""),
            privacy_status=privacy_status,
            access_token=access_token,
        )
        return ToolResult(success=True, data={"playlist": playlist.to_dict()})


class DeletePlaylistTool(Tool):
    name = "youtube.playlist.delete"
    description = "Permanently delete a playlist from the connected account."
    input_schema = {"playlist_id": {"type": "string", "required": True}, "access_token": {"type": "string", "required": True}}
    output_schema = {"deleted": {"type": "boolean"}}
    permissions = [_YOUTUBE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        playlist_id = (arguments.get("playlist_id") or "").strip()
        if not playlist_id:
            raise ValidationError("`playlist_id` is required for youtube.playlist.delete.")
        access_token = _require_access_token(arguments)

        self._client.delete_playlist(playlist_id=playlist_id, access_token=access_token)
        return ToolResult(success=True, data={"deleted": True, "playlist_id": playlist_id})


class AddPlaylistItemTool(Tool):
    name = "youtube.playlist.add_video"
    description = "Add a video to a playlist, optionally at a specific position."
    input_schema = {
        "playlist_id": {"type": "string", "required": True},
        "video_id": {"type": "string", "required": True},
        "position": {"type": "integer", "required": False},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"item": {"type": "object"}}
    permissions = [_YOUTUBE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        playlist_id = (arguments.get("playlist_id") or "").strip()
        video_id = (arguments.get("video_id") or "").strip()
        if not playlist_id or not video_id:
            raise ValidationError("`playlist_id` and `video_id` are required for youtube.playlist.add_video.")
        access_token = _require_access_token(arguments)

        item = self._client.add_playlist_item(
            playlist_id=playlist_id, video_id=video_id, position=arguments.get("position"), access_token=access_token
        )
        return ToolResult(success=True, data={"item": item.to_dict()})


class RemovePlaylistItemTool(Tool):
    name = "youtube.playlist.remove_video"
    description = "Remove a video from a playlist by its playlist-item ID."
    input_schema = {"playlist_item_id": {"type": "string", "required": True}, "access_token": {"type": "string", "required": True}}
    output_schema = {"removed": {"type": "boolean"}}
    permissions = [_YOUTUBE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        playlist_item_id = (arguments.get("playlist_item_id") or "").strip()
        if not playlist_item_id:
            raise ValidationError("`playlist_item_id` is required for youtube.playlist.remove_video.")
        access_token = _require_access_token(arguments)

        self._client.remove_playlist_item(playlist_item_id=playlist_item_id, access_token=access_token)
        return ToolResult(success=True, data={"removed": True, "playlist_item_id": playlist_item_id})


class ReorderPlaylistItemTool(Tool):
    name = "youtube.playlist.reorder_video"
    description = "Move a video already in a playlist to a new (0-based) position."
    input_schema = {
        "playlist_item_id": {"type": "string", "required": True},
        "playlist_id": {"type": "string", "required": True},
        "video_id": {"type": "string", "required": True},
        "position": {"type": "integer", "required": True},
        "access_token": {"type": "string", "required": True},
    }
    output_schema = {"item": {"type": "object"}}
    permissions = [_YOUTUBE_SCOPE]
    requires_confirmation = True

    def __init__(self, *, client: YouTubeApiClient) -> None:
        self._client = client

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        playlist_item_id = (arguments.get("playlist_item_id") or "").strip()
        playlist_id = (arguments.get("playlist_id") or "").strip()
        video_id = (arguments.get("video_id") or "").strip()
        position = arguments.get("position")
        if not playlist_item_id or not playlist_id or not video_id or position is None:
            raise ValidationError(
                "`playlist_item_id`, `playlist_id`, `video_id`, and `position` are all required "
                "for youtube.playlist.reorder_video."
            )
        access_token = _require_access_token(arguments)

        item = self._client.reorder_playlist_item(
            playlist_item_id=playlist_item_id,
            playlist_id=playlist_id,
            video_id=video_id,
            position=int(position),
            access_token=access_token,
        )
        return ToolResult(success=True, data={"item": item.to_dict()})
