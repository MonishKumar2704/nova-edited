"""
YouTube tools (search, player control, playlists, channel, account).

Phase 3 (YouTube API Foundation) adds `youtube.search`, `youtube.get_video`,
and `youtube.list_channels`, all backed by the official YouTube Data API v3
(`app.integrations.youtube_api`). Player control (Phase 4, client-side -
see `app/static/js/youtube-player.js`) and playlist/account/video
management (Phase 5) are added the same way: a new `Tool` subclass here,
registered via `register_youtube_tools`, with zero changes to the
orchestrator or the registry itself (master spec section 7).
"""

from __future__ import annotations

from app.core.cache import TTLCache
from app.integrations.youtube_api import YouTubeApiClient
from app.tools.registry import ToolRegistry
from app.tools.youtube.channel import MyUploadsTool
from app.tools.youtube.get_video import GetVideoTool
from app.tools.youtube.list_channels import ListMyChannelTool
from app.tools.youtube.playlists import (
    AddPlaylistItemTool,
    CreatePlaylistTool,
    DeletePlaylistTool,
    GetPlaylistTool,
    ListMyPlaylistsTool,
    RemovePlaylistItemTool,
    ReorderPlaylistItemTool,
    UpdatePlaylistTool,
)
from app.tools.youtube.search_videos import SearchVideosTool
from app.tools.youtube.video_management import DeleteVideoTool, GetVideoRatingTool, RateVideoTool, UpdateVideoTool


def register_youtube_tools(
    registry: ToolRegistry,
    *,
    client: YouTubeApiClient,
    search_cache: TTLCache | None = None,
) -> None:
    # Phase 3: public/search data
    registry.register(SearchVideosTool(client=client, cache=search_cache))
    registry.register(GetVideoTool(client=client, cache=search_cache))
    registry.register(ListMyChannelTool(client=client))

    # Phase 5: playlists
    registry.register(ListMyPlaylistsTool(client=client))
    registry.register(GetPlaylistTool(client=client))
    registry.register(CreatePlaylistTool(client=client))
    registry.register(UpdatePlaylistTool(client=client))
    registry.register(DeletePlaylistTool(client=client))
    registry.register(AddPlaylistItemTool(client=client))
    registry.register(RemovePlaylistItemTool(client=client))
    registry.register(ReorderPlaylistItemTool(client=client))

    # Phase 5: account/channel + video management
    registry.register(MyUploadsTool(client=client))
    registry.register(UpdateVideoTool(client=client))
    registry.register(DeleteVideoTool(client=client))
    registry.register(RateVideoTool(client=client))
    registry.register(GetVideoRatingTool(client=client))
