from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class YouTubeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    max_results: int = Field(10, ge=1, le=25)
    order: str = Field("relevance")


# --- Phase 5: playlists / video management -----------------------------


class PlaylistCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=150)
    description: str = Field("", max_length=5000)
    privacy_status: Literal["private", "public", "unlisted"] = "private"


class PlaylistUpdateRequest(PlaylistCreateRequest):
    pass


class PlaylistAddVideoRequest(BaseModel):
    video_id: str = Field(..., min_length=1, max_length=32)
    position: int | None = Field(None, ge=0)


class PlaylistReorderRequest(BaseModel):
    playlist_item_id: str = Field(..., min_length=1)
    video_id: str = Field(..., min_length=1, max_length=32)
    position: int = Field(..., ge=0)


class VideoUpdateRequest(BaseModel):
    title: str | None = Field(None, max_length=100)
    description: str | None = Field(None, max_length=5000)
    tags: list[str] | None = None


class VideoRateRequest(BaseModel):
    rating: Literal["like", "dislike", "none"]
