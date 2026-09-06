"""
`/api/v1/youtube/...` (master spec section 45 API DESIGN, Phase 3).

Route handlers stay thin (master spec section 5): parse/validate the
request, resolve an OAuth access token if one is available for this
session, call the appropriate tool via the registry, serialize the
`ToolResult`. All YouTube Data API logic lives in
`app.integrations.youtube_api` / `app.tools.youtube.*`.
"""

from __future__ import annotations

from flask import Blueprint, g, jsonify, request
from pydantic import ValidationError as PydanticValidationError

from app.api.v1._google_auth_helpers import registry as _registry
from app.api.v1._google_auth_helpers import require_access_token, resolve_access_token
from app.core.errors import AuthenticationError, ValidationError
from app.schemas.youtube import (
    PlaylistAddVideoRequest,
    PlaylistCreateRequest,
    PlaylistReorderRequest,
    PlaylistUpdateRequest,
    VideoRateRequest,
    VideoUpdateRequest,
    YouTubeSearchRequest,
)
from app.services.dynamic_ui import build_playlist_cards, build_playlist_item_cards, build_video_cards

youtube_bp = Blueprint("youtube", __name__)


def _optional_access_token() -> str | None:
    """Best-effort access token for the current session, or None.

    YouTube search/get_video work fine without one (falls back to the
    configured API key) - a connected account is a bonus (raises quota
    limits, unlocks account-specific ordering) but never required.
    """
    return resolve_access_token()


def _required_access_token() -> str:
    """Access token for the current session, or a clear 401 (Phase 5 writes).

    Unlike `_optional_access_token`, playlist/video-management endpoints
    have no API-key fallback (YouTube has no such path for writes - see
    `YouTubeApiClient._write`), so a missing/invalid session is always an
    error here rather than a silent `None`.
    """
    return require_access_token(
        "This action requires a connected Google account. Connect one via /api/v1/auth/google/connect."
    )


@youtube_bp.route("/search", methods=["GET"])
def search():
    try:
        req = YouTubeSearchRequest(
            query=request.args.get("q", ""),
            max_results=int(request.args.get("max_results", 10)),
            order=request.args.get("order", "relevance"),
        )
    except (PydanticValidationError, ValueError) as exc:
        raise ValidationError("Invalid search request.", details={"errors": str(exc)}) from exc

    tool = _registry().get("youtube.search")
    result = tool.execute(
        {
            "query": req.query,
            "max_results": req.max_results,
            "order": req.order,
            "access_token": _optional_access_token(),
        }
    )
    results = result.data.get("results", [])
    return jsonify(
        {
            "success": result.success,
            "query": req.query,
            "results": results,
            # Dynamic-UI-ready cards (master spec section 10, Phase 4): same
            # video data as `results`, wrapped with a `type` + `actions` list
            # the frontend renders generically. `results` is kept as-is for
            # any existing caller that only wants the raw video dicts.
            "cards": build_video_cards(results),
            "cached": result.data.get("cached", False),
            "request_id": getattr(g, "request_id", None),
        }
    )


@youtube_bp.route("/videos/<video_id>", methods=["GET"])
def get_video(video_id: str):
    tool = _registry().get("youtube.get_video")
    result = tool.execute({"video_id": video_id, "access_token": _optional_access_token()})
    results = result.data.get("results", [])
    return jsonify(
        {
            "success": result.success,
            "results": results,
            "cards": build_video_cards(results),
            "cached": result.data.get("cached", False),
            "request_id": getattr(g, "request_id", None),
        }
    )


@youtube_bp.route("/channels/mine", methods=["GET"])
def my_channel():
    access_token = _optional_access_token()
    if not access_token:
        raise AuthenticationError(
            "youtube.list_channels requires a connected Google account. Connect one via /api/v1/auth/google/connect."
        )

    tool = _registry().get("youtube.list_channels")
    result = tool.execute({"access_token": access_token})
    return jsonify(
        {
            "success": result.success,
            "channel": result.data.get("channel"),
            "request_id": getattr(g, "request_id", None),
        }
    )


@youtube_bp.route("/channels/mine/uploads", methods=["GET"])
def my_uploads():
    access_token = _required_access_token()
    tool = _registry().get("youtube.channel.my_uploads")
    result = tool.execute(
        {
            "access_token": access_token,
            "max_results": int(request.args.get("max_results", 25)),
            "page_token": request.args.get("page_token"),
        }
    )
    uploads = result.data.get("uploads", [])
    return jsonify(
        {
            "success": result.success,
            "uploads": uploads,
            "cards": build_playlist_item_cards(uploads),
            "next_page_token": result.data.get("next_page_token"),
            "request_id": getattr(g, "request_id", None),
        }
    )


# --- Playlists (Phase 5) -------------------------------------------------


@youtube_bp.route("/playlists", methods=["GET"])
def list_playlists():
    access_token = _required_access_token()
    tool = _registry().get("youtube.playlist.list")
    result = tool.execute(
        {
            "access_token": access_token,
            "max_results": int(request.args.get("max_results", 25)),
            "page_token": request.args.get("page_token"),
        }
    )
    playlists = result.data.get("playlists", [])
    return jsonify(
        {
            "success": result.success,
            "playlists": playlists,
            "cards": build_playlist_cards(playlists),
            "next_page_token": result.data.get("next_page_token"),
            "request_id": getattr(g, "request_id", None),
        }
    )


@youtube_bp.route("/playlists", methods=["POST"])
def create_playlist():
    payload = request.get_json(silent=True) or {}
    try:
        req = PlaylistCreateRequest(**payload)
    except PydanticValidationError as exc:
        raise ValidationError("Invalid playlist create request.", details={"errors": str(exc)}) from exc

    tool = _registry().get("youtube.playlist.create")
    result = tool.execute(
        {
            "title": req.title,
            "description": req.description,
            "privacy_status": req.privacy_status,
            "access_token": _required_access_token(),
        }
    )
    return jsonify({"success": result.success, "playlist": result.data.get("playlist"), "request_id": getattr(g, "request_id", None)}), 201


@youtube_bp.route("/playlists/<playlist_id>", methods=["GET"])
def get_playlist(playlist_id: str):
    tool = _registry().get("youtube.playlist.get")
    result = tool.execute(
        {
            "playlist_id": playlist_id,
            "access_token": _optional_access_token(),
            "max_results": int(request.args.get("max_results", 25)),
            "page_token": request.args.get("page_token"),
        }
    )
    items = result.data.get("items", [])
    return jsonify(
        {
            "success": result.success,
            "playlist": result.data.get("playlist"),
            "items": items,
            "cards": build_playlist_item_cards(items),
            "next_page_token": result.data.get("next_page_token"),
            "request_id": getattr(g, "request_id", None),
        }
    )


@youtube_bp.route("/playlists/<playlist_id>", methods=["PUT"])
def update_playlist(playlist_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        req = PlaylistUpdateRequest(**payload)
    except PydanticValidationError as exc:
        raise ValidationError("Invalid playlist update request.", details={"errors": str(exc)}) from exc

    tool = _registry().get("youtube.playlist.update")
    result = tool.execute(
        {
            "playlist_id": playlist_id,
            "title": req.title,
            "description": req.description,
            "privacy_status": req.privacy_status,
            "access_token": _required_access_token(),
        }
    )
    return jsonify({"success": result.success, "playlist": result.data.get("playlist"), "request_id": getattr(g, "request_id", None)})


@youtube_bp.route("/playlists/<playlist_id>", methods=["DELETE"])
def delete_playlist(playlist_id: str):
    tool = _registry().get("youtube.playlist.delete")
    result = tool.execute({"playlist_id": playlist_id, "access_token": _required_access_token()})
    return jsonify({"success": result.success, "deleted": result.data.get("deleted", False), "request_id": getattr(g, "request_id", None)})


@youtube_bp.route("/playlists/<playlist_id>/items", methods=["POST"])
def add_playlist_item(playlist_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        req = PlaylistAddVideoRequest(**payload)
    except PydanticValidationError as exc:
        raise ValidationError("Invalid add-video request.", details={"errors": str(exc)}) from exc

    tool = _registry().get("youtube.playlist.add_video")
    result = tool.execute(
        {
            "playlist_id": playlist_id,
            "video_id": req.video_id,
            "position": req.position,
            "access_token": _required_access_token(),
        }
    )
    return jsonify({"success": result.success, "item": result.data.get("item"), "request_id": getattr(g, "request_id", None)}), 201


@youtube_bp.route("/playlists/<playlist_id>/items/<playlist_item_id>", methods=["DELETE"])
def remove_playlist_item(playlist_id: str, playlist_item_id: str):
    tool = _registry().get("youtube.playlist.remove_video")
    result = tool.execute({"playlist_item_id": playlist_item_id, "access_token": _required_access_token()})
    return jsonify({"success": result.success, "removed": result.data.get("removed", False), "request_id": getattr(g, "request_id", None)})


@youtube_bp.route("/playlists/<playlist_id>/items/<playlist_item_id>/reorder", methods=["POST"])
def reorder_playlist_item(playlist_id: str, playlist_item_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        req = PlaylistReorderRequest(**payload)
    except PydanticValidationError as exc:
        raise ValidationError("Invalid reorder request.", details={"errors": str(exc)}) from exc

    tool = _registry().get("youtube.playlist.reorder_video")
    result = tool.execute(
        {
            "playlist_item_id": playlist_item_id,
            "playlist_id": playlist_id,
            "video_id": req.video_id,
            "position": req.position,
            "access_token": _required_access_token(),
        }
    )
    return jsonify({"success": result.success, "item": result.data.get("item"), "request_id": getattr(g, "request_id", None)})


# --- Video management (Phase 5) ------------------------------------------


@youtube_bp.route("/videos/<video_id>", methods=["PUT"])
def update_video(video_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        req = VideoUpdateRequest(**payload)
    except PydanticValidationError as exc:
        raise ValidationError("Invalid video update request.", details={"errors": str(exc)}) from exc

    tool = _registry().get("youtube.video.update")
    result = tool.execute(
        {
            "video_id": video_id,
            "title": req.title,
            "description": req.description,
            "tags": req.tags,
            "access_token": _required_access_token(),
        }
    )
    return jsonify({"success": result.success, "video": result.data.get("video"), "request_id": getattr(g, "request_id", None)})


@youtube_bp.route("/videos/<video_id>", methods=["DELETE"])
def delete_video(video_id: str):
    tool = _registry().get("youtube.video.delete")
    result = tool.execute({"video_id": video_id, "access_token": _required_access_token()})
    return jsonify({"success": result.success, "deleted": result.data.get("deleted", False), "request_id": getattr(g, "request_id", None)})


@youtube_bp.route("/videos/<video_id>/rating", methods=["GET"])
def get_video_rating(video_id: str):
    tool = _registry().get("youtube.video.get_rating")
    result = tool.execute({"video_id": video_id, "access_token": _required_access_token()})
    return jsonify({"success": result.success, "rating": result.data.get("rating"), "request_id": getattr(g, "request_id", None)})


@youtube_bp.route("/videos/<video_id>/rating", methods=["PUT"])
def rate_video(video_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        req = VideoRateRequest(**payload)
    except PydanticValidationError as exc:
        raise ValidationError("Invalid rate request.", details={"errors": str(exc)}) from exc

    tool = _registry().get("youtube.video.rate")
    result = tool.execute({"video_id": video_id, "rating": req.rating, "access_token": _required_access_token()})
    return jsonify({"success": result.success, "rating": result.data.get("rating"), "request_id": getattr(g, "request_id", None)})
