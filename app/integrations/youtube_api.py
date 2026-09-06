"""
Official YouTube Data API v3 client (master spec sections 18-19, 59:
official APIs instead of fragile HTML scraping).

Implemented as a thin REST wrapper with `requests`, matching the style of
`app.integrations.google_oauth` - no `google-api-python-client` dependency
is pulled in for the handful of endpoints Phase 3 needs. This module has
ZERO Flask/session/tool-registry knowledge; it only knows how to talk to
YouTube. Orchestration (caching, argument validation, ToolResult shaping)
lives in `app.tools.youtube.*`.

Auth model
----------
* `search()` and `get_videos()` cover public data and work with either an
  API key (`YOUTUBE_API_KEY`, no user has to connect anything - the
  default/free path) or a user's OAuth access token, whichever is
  supplied.
* `get_my_channel()` is inherently account-specific (`mine=true`) and
  requires an OAuth access token with the `youtube.readonly` scope -
  there is no API-key path for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from app.core.errors import (
    AuthenticationError,
    NetworkError,
    NotFoundError,
    NovaError,
    RateLimitError,
    TimeoutErrorNova,
    ValidationError,
)

API_BASE = "https://www.googleapis.com/youtube/v3"
_REQUEST_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class VideoSummary:
    video_id: str
    title: str
    channel_title: str
    description: str
    thumbnail_url: str | None
    published_at: str | None
    url: str

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "channel_title": self.channel_title,
            "description": self.description,
            "thumbnail_url": self.thumbnail_url,
            "published_at": self.published_at,
            "url": self.url,
        }


@dataclass(frozen=True)
class VideoDetails(VideoSummary):
    duration_iso8601: str | None = None
    view_count: int | None = None
    like_count: int | None = None
    # Only populated by `get_videos()` (which requests `contentDetails` +
    # `statistics`); needed by `update_video()` since `videos.update`
    # replaces the whole `snippet` and would otherwise silently blank the
    # category out on a title/description-only edit (Phase 5).
    category_id: str | None = None

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "duration": self.duration_iso8601,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "category_id": self.category_id,
        }


@dataclass(frozen=True)
class ChannelSummary:
    channel_id: str
    title: str
    description: str
    thumbnail_url: str | None
    subscriber_count: int | None
    video_count: int | None
    # Playlist ID for the channel's own uploads (Phase 5: `channels.list`
    # `contentDetails.relatedPlaylists.uploads`). Used to list "my uploaded
    # videos" via the ordinary playlistItems endpoint instead of a
    # dedicated (nonexistent) "my videos" API.
    uploads_playlist_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "title": self.title,
            "description": self.description,
            "thumbnail_url": self.thumbnail_url,
            "subscriber_count": self.subscriber_count,
            "video_count": self.video_count,
            "uploads_playlist_id": self.uploads_playlist_id,
        }


@dataclass(frozen=True)
class PlaylistSummary:
    playlist_id: str
    title: str
    description: str
    thumbnail_url: str | None
    privacy_status: str | None
    item_count: int | None

    def to_dict(self) -> dict:
        return {
            "playlist_id": self.playlist_id,
            "title": self.title,
            "description": self.description,
            "thumbnail_url": self.thumbnail_url,
            "privacy_status": self.privacy_status,
            "item_count": self.item_count,
        }


@dataclass(frozen=True)
class PlaylistItem:
    """One entry in a playlist (`playlistItems` resource).

    Distinct from `VideoSummary`: `playlist_item_id` (needed to remove or
    reorder the entry) and `position` only exist in this context, not on
    the underlying video itself.
    """

    playlist_item_id: str
    playlist_id: str
    video_id: str
    title: str
    channel_title: str
    thumbnail_url: str | None
    position: int | None

    def to_dict(self) -> dict:
        return {
            "playlist_item_id": self.playlist_item_id,
            "playlist_id": self.playlist_id,
            "video_id": self.video_id,
            "title": self.title,
            "channel_title": self.channel_title,
            "thumbnail_url": self.thumbnail_url,
            "position": self.position,
            "url": f"https://www.youtube.com/watch?v={self.video_id}",
        }


class YouTubeApiClient:
    def __init__(self, *, api_key: str = "") -> None:
        self._api_key = api_key

    def is_configured(self) -> bool:
        """True if public (API-key) access is available.

        OAuth-only calls (`get_my_channel`) don't depend on this - a
        connected Google account is enough even with no API key set.
        """
        return bool(self._api_key)

    def search(
        self,
        *,
        query: str,
        max_results: int = 10,
        order: str = "relevance",
        access_token: str | None = None,
    ) -> list[VideoSummary]:
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max(1, min(max_results, 25)),
            "order": order,
            "safeSearch": "moderate",
        }
        body = self._get("/search", params=params, access_token=access_token)
        results = []
        for item in body.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            if not video_id:
                continue
            results.append(self._summary_from_snippet(video_id, item.get("snippet", {})))
        return results

    def get_videos(self, *, video_ids: list[str], access_token: str | None = None) -> list[VideoDetails]:
        if not video_ids:
            return []
        params = {
            "part": "snippet,contentDetails,statistics",
            "id": ",".join(video_ids[:50]),
        }
        body = self._get("/videos", params=params, access_token=access_token)
        results = []
        for item in body.get("items", []):
            snippet = item.get("snippet", {})
            content_details = item.get("contentDetails", {})
            statistics = item.get("statistics", {})
            summary = self._summary_from_snippet(item.get("id", ""), snippet)
            results.append(
                VideoDetails(
                    **summary.__dict__,
                    duration_iso8601=content_details.get("duration"),
                    view_count=_safe_int(statistics.get("viewCount")),
                    like_count=_safe_int(statistics.get("likeCount")),
                    category_id=snippet.get("categoryId"),
                )
            )
        return results

    def get_my_channel(self, *, access_token: str) -> ChannelSummary | None:
        if not access_token:
            raise AuthenticationError("A Google access token is required to look up the connected account's channel.")
        params = {"part": "snippet,statistics,contentDetails", "mine": "true"}
        body = self._get("/channels", params=params, access_token=access_token)
        items = body.get("items", [])
        if not items:
            return None
        return self._channel_from_item(items[0])

    # -- Playlists (Phase 5) -----------------------------------------
    # https://developers.google.com/youtube/v3/docs/playlists and
    # .../playlistItems - reordering has no dedicated endpoint; it's done
    # by `playlistItems.update` with a different `position` (master spec
    # Phase 5: "reorder where officially supported").

    def list_my_playlists(
        self, *, access_token: str, max_results: int = 25, page_token: str | None = None
    ) -> tuple[list[PlaylistSummary], str | None]:
        if not access_token:
            raise AuthenticationError("A Google access token is required to list your playlists.")
        params: dict = {"part": "snippet,contentDetails,status", "mine": "true", "maxResults": max(1, min(max_results, 50))}
        if page_token:
            params["pageToken"] = page_token
        body = self._get("/playlists", params=params, access_token=access_token)
        playlists = [self._playlist_from_item(item) for item in body.get("items", [])]
        return playlists, body.get("nextPageToken")

    def get_playlist(self, *, playlist_id: str, access_token: str | None = None) -> PlaylistSummary | None:
        params = {"part": "snippet,contentDetails,status", "id": playlist_id}
        body = self._get("/playlists", params=params, access_token=access_token)
        items = body.get("items", [])
        return self._playlist_from_item(items[0]) if items else None

    def create_playlist(
        self, *, title: str, description: str = "", privacy_status: str = "private", access_token: str
    ) -> PlaylistSummary:
        if not access_token:
            raise AuthenticationError("A Google access token is required to create a playlist.")
        payload = {
            "snippet": {"title": title, "description": description},
            "status": {"privacyStatus": privacy_status},
        }
        body = self._post("/playlists", params={"part": "snippet,status"}, json_body=payload, access_token=access_token)
        return self._playlist_from_item(body)

    def update_playlist(
        self,
        *,
        playlist_id: str,
        title: str,
        description: str = "",
        privacy_status: str = "private",
        access_token: str,
    ) -> PlaylistSummary:
        if not access_token:
            raise AuthenticationError("A Google access token is required to update a playlist.")
        payload = {
            "id": playlist_id,
            "snippet": {"title": title, "description": description},
            "status": {"privacyStatus": privacy_status},
        }
        body = self._put("/playlists", params={"part": "snippet,status"}, json_body=payload, access_token=access_token)
        return self._playlist_from_item(body)

    def delete_playlist(self, *, playlist_id: str, access_token: str) -> None:
        if not access_token:
            raise AuthenticationError("A Google access token is required to delete a playlist.")
        self._delete("/playlists", params={"id": playlist_id}, access_token=access_token)

    def list_playlist_items(
        self, *, playlist_id: str, max_results: int = 25, page_token: str | None = None, access_token: str | None = None
    ) -> tuple[list[PlaylistItem], str | None]:
        params: dict = {"part": "snippet", "playlistId": playlist_id, "maxResults": max(1, min(max_results, 50))}
        if page_token:
            params["pageToken"] = page_token
        body = self._get("/playlistItems", params=params, access_token=access_token)
        items = [self._playlist_item_from_item(item) for item in body.get("items", [])]
        return items, body.get("nextPageToken")

    def add_playlist_item(
        self, *, playlist_id: str, video_id: str, position: int | None = None, access_token: str
    ) -> PlaylistItem:
        if not access_token:
            raise AuthenticationError("A Google access token is required to modify a playlist.")
        resource_id: dict = {"kind": "youtube#video", "videoId": video_id}
        snippet: dict = {"playlistId": playlist_id, "resourceId": resource_id}
        if position is not None:
            snippet["position"] = position
        payload = {"snippet": snippet}
        body = self._post("/playlistItems", params={"part": "snippet"}, json_body=payload, access_token=access_token)
        return self._playlist_item_from_item(body)

    def remove_playlist_item(self, *, playlist_item_id: str, access_token: str) -> None:
        if not access_token:
            raise AuthenticationError("A Google access token is required to modify a playlist.")
        self._delete("/playlistItems", params={"id": playlist_item_id}, access_token=access_token)

    def reorder_playlist_item(
        self, *, playlist_item_id: str, playlist_id: str, video_id: str, position: int, access_token: str
    ) -> PlaylistItem:
        """Move an existing playlist entry to a new `position` (0-based)."""
        if not access_token:
            raise AuthenticationError("A Google access token is required to modify a playlist.")
        payload = {
            "id": playlist_item_id,
            "snippet": {
                "playlistId": playlist_id,
                "position": position,
                "resourceId": {"kind": "youtube#video", "videoId": video_id},
            },
        }
        body = self._put("/playlistItems", params={"part": "snippet"}, json_body=payload, access_token=access_token)
        return self._playlist_item_from_item(body)

    def get_my_uploads(
        self, *, access_token: str, max_results: int = 25, page_token: str | None = None
    ) -> tuple[list[PlaylistItem], str | None]:
        """The connected account's own uploaded videos, via its uploads playlist."""
        channel = self.get_my_channel(access_token=access_token)
        if channel is None or not channel.uploads_playlist_id:
            return [], None
        return self.list_playlist_items(
            playlist_id=channel.uploads_playlist_id,
            max_results=max_results,
            page_token=page_token,
            access_token=access_token,
        )

    # -- Video management (Phase 5) -----------------------------------

    def update_video(
        self,
        *,
        video_id: str,
        title: str,
        description: str,
        category_id: str,
        access_token: str,
        tags: list[str] | None = None,
    ) -> VideoDetails:
        """Update a video's own metadata.

        `videos.update` replaces the entire `snippet`, so `title`,
        `description`, and `category_id` must all be supplied (callers
        should fetch current values first for a partial-looking edit -
        see `VideoUpdateTool`).
        """
        if not access_token:
            raise AuthenticationError("A Google access token is required to update a video.")
        snippet: dict = {"title": title, "description": description, "categoryId": category_id}
        if tags is not None:
            snippet["tags"] = tags
        payload = {"id": video_id, "snippet": snippet}
        body = self._put("/videos", params={"part": "snippet"}, json_body=payload, access_token=access_token)
        snippet_out = body.get("snippet", {})
        summary = self._summary_from_snippet(body.get("id", video_id), snippet_out)
        return VideoDetails(**summary.__dict__)

    def delete_video(self, *, video_id: str, access_token: str) -> None:
        if not access_token:
            raise AuthenticationError("A Google access token is required to delete a video.")
        self._delete("/videos", params={"id": video_id}, access_token=access_token)

    def rate_video(self, *, video_id: str, rating: str, access_token: str) -> None:
        """`rating` is one of YouTube's own values: like | dislike | none."""
        if not access_token:
            raise AuthenticationError("A Google access token is required to rate a video.")
        if rating not in {"like", "dislike", "none"}:
            raise ValidationError("`rating` must be one of: like, dislike, none.")
        self._post(
            "/videos/rate",
            params={"id": video_id, "rating": rating},
            json_body=None,
            access_token=access_token,
            expect_json=False,
        )

    def get_video_rating(self, *, video_id: str, access_token: str) -> str | None:
        if not access_token:
            raise AuthenticationError("A Google access token is required to read a video rating.")
        body = self._get("/videos/getRating", params={"id": video_id}, access_token=access_token)
        items = body.get("items", [])
        return items[0].get("rating") if items else None

    # -- internal ---------------------------------------------------

    def _channel_from_item(self, item: dict) -> ChannelSummary:
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})
        content_details = item.get("contentDetails", {})
        uploads_playlist_id = content_details.get("relatedPlaylists", {}).get("uploads")
        return ChannelSummary(
            channel_id=item.get("id", ""),
            title=snippet.get("title", ""),
            description=snippet.get("description", ""),
            thumbnail_url=_best_thumbnail(snippet.get("thumbnails")),
            subscriber_count=_safe_int(statistics.get("subscriberCount")),
            video_count=_safe_int(statistics.get("videoCount")),
            uploads_playlist_id=uploads_playlist_id,
        )

    def _playlist_from_item(self, item: dict) -> PlaylistSummary:
        snippet = item.get("snippet", {})
        status = item.get("status", {})
        content_details = item.get("contentDetails", {})
        return PlaylistSummary(
            playlist_id=item.get("id", ""),
            title=snippet.get("title", ""),
            description=snippet.get("description", ""),
            thumbnail_url=_best_thumbnail(snippet.get("thumbnails")),
            privacy_status=status.get("privacyStatus"),
            item_count=_safe_int(content_details.get("itemCount")),
        )

    def _playlist_item_from_item(self, item: dict) -> PlaylistItem:
        snippet = item.get("snippet", {})
        resource_id = snippet.get("resourceId", {})
        return PlaylistItem(
            playlist_item_id=item.get("id", ""),
            playlist_id=snippet.get("playlistId", ""),
            video_id=resource_id.get("videoId", ""),
            title=snippet.get("title", ""),
            channel_title=snippet.get("channelTitle", ""),
            thumbnail_url=_best_thumbnail(snippet.get("thumbnails")),
            position=_safe_int(snippet.get("position")),
        )

    def _summary_from_snippet(self, video_id: str, snippet: dict) -> VideoSummary:
        return VideoSummary(
            video_id=video_id,
            title=snippet.get("title", ""),
            channel_title=snippet.get("channelTitle", ""),
            description=snippet.get("description", ""),
            thumbnail_url=_best_thumbnail(snippet.get("thumbnails")),
            published_at=snippet.get("publishedAt"),
            url=f"https://www.youtube.com/watch?v={video_id}",
        )

    def _get(self, path: str, *, params: dict, access_token: str | None) -> dict:
        headers = {}
        request_params = dict(params)
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        elif self._api_key:
            request_params["key"] = self._api_key
        else:
            raise ValidationError(
                "YouTube API is not configured: set YOUTUBE_API_KEY, or connect a Google account, "
                "before calling YouTube tools."
            )

        try:
            resp = requests.get(f"{API_BASE}{path}", params=request_params, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS)
        except requests.Timeout as exc:
            raise TimeoutErrorNova("Timed out talking to the YouTube Data API.") from exc
        except requests.RequestException as exc:
            raise NetworkError("Could not reach the YouTube Data API.") from exc

        if resp.status_code == 401:
            raise AuthenticationError("YouTube rejected the request credentials (expired/invalid token or key).")
        if resp.status_code == 403:
            raise _classify_403(resp)
        if resp.status_code == 400:
            raise ValidationError(f"YouTube API rejected the request: {_extract_error_message(resp)}")
        if not resp.ok:
            raise NetworkError(f"YouTube Data API returned HTTP {resp.status_code}.")

        return resp.json()

    def _post(
        self,
        path: str,
        *,
        params: dict,
        json_body: dict | None,
        access_token: str | None,
        expect_json: bool = True,
    ) -> dict:
        return self._write("post", path, params=params, json_body=json_body, access_token=access_token, expect_json=expect_json)

    def _put(self, path: str, *, params: dict, json_body: dict | None, access_token: str | None) -> dict:
        return self._write("put", path, params=params, json_body=json_body, access_token=access_token, expect_json=True)

    def _delete(self, path: str, *, params: dict, access_token: str | None) -> None:
        self._write("delete", path, params=params, json_body=None, access_token=access_token, expect_json=False)

    def _write(
        self,
        method: str,
        path: str,
        *,
        params: dict,
        json_body: dict | None,
        access_token: str | None,
        expect_json: bool,
    ) -> dict:
        if not access_token:
            # Every write endpoint requires OAuth (master spec section 18:
            # no API-key-only mutation path exists on YouTube's API).
            raise AuthenticationError("A Google access token is required for this YouTube operation.")
        headers = {"Authorization": f"Bearer {access_token}"}

        try:
            resp = requests.request(
                method,
                f"{API_BASE}{path}",
                params=params,
                json=json_body,
                headers=headers,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.Timeout as exc:
            raise TimeoutErrorNova("Timed out talking to the YouTube Data API.") from exc
        except requests.RequestException as exc:
            raise NetworkError("Could not reach the YouTube Data API.") from exc

        if resp.status_code == 401:
            raise AuthenticationError("YouTube rejected the request credentials (expired/invalid token).")
        if resp.status_code == 403:
            raise _classify_403(resp)
        if resp.status_code == 404:
            raise NotFoundError(f"YouTube resource not found: {_extract_error_message(resp)}")
        if resp.status_code == 400:
            raise ValidationError(f"YouTube API rejected the request: {_extract_error_message(resp)}")
        if not resp.ok:
            raise NetworkError(f"YouTube Data API returned HTTP {resp.status_code}.")

        if not expect_json or not resp.content:
            return {}
        return resp.json()


def _classify_403(resp: requests.Response) -> NovaError:
    message = _extract_error_message(resp)
    lowered = message.lower()
    if "quota" in lowered:
        return RateLimitError(f"YouTube API quota exceeded: {message}")
    return AuthenticationError(f"YouTube API access forbidden: {message}")


def _extract_error_message(resp: requests.Response) -> str:
    try:
        body = resp.json()
        return body.get("error", {}).get("message", resp.text[:200])
    except ValueError:
        return resp.text[:200]


def _best_thumbnail(thumbnails: dict | None) -> str | None:
    if not thumbnails:
        return None
    for key in ("high", "medium", "default"):
        if key in thumbnails:
            return thumbnails[key].get("url")
    return None


def _safe_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
