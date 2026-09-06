from unittest.mock import MagicMock

import pytest

from app.core.cache import TTLCache
from app.core.errors import AuthenticationError, ValidationError
from app.integrations.youtube_api import ChannelSummary, VideoDetails, VideoSummary
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

SUMMARY = VideoSummary(
    video_id="vid1",
    title="Python Tutorial",
    channel_title="Code Channel",
    description="desc",
    thumbnail_url=None,
    published_at=None,
    url="https://www.youtube.com/watch?v=vid1",
)


def test_search_tool_requires_query():
    tool = SearchVideosTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({})


def test_search_tool_rejects_invalid_order():
    tool = SearchVideosTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"query": "python", "order": "not-a-real-order"})


def test_search_tool_returns_results_and_caches():
    client = MagicMock()
    client.search.return_value = [SUMMARY]
    cache = TTLCache(ttl_seconds=60)
    tool = SearchVideosTool(client=client, cache=cache)

    result = tool.execute({"query": "python"})
    assert result.success is True
    assert result.data["results"][0]["video_id"] == "vid1"
    assert result.data["cached"] is False

    # Second identical call should be served from cache, no second API call.
    result2 = tool.execute({"query": "python"})
    assert result2.data["cached"] is True
    client.search.assert_called_once()


def test_get_video_tool_merges_single_and_list_ids():
    client = MagicMock()
    details = VideoDetails(**SUMMARY.__dict__, duration_iso8601="PT5M", view_count=1, like_count=1)
    client.get_videos.return_value = [details]
    tool = GetVideoTool(client=client)

    result = tool.execute({"video_id": "vid1", "video_ids": ["vid1", "vid2"]})
    assert result.success is True
    called_ids = client.get_videos.call_args.kwargs["video_ids"]
    assert called_ids == ["vid1", "vid2"]  # de-duped, order preserved


def test_get_video_tool_requires_at_least_one_id():
    tool = GetVideoTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({})


def test_list_channels_tool_requires_access_token():
    tool = ListMyChannelTool(client=MagicMock())
    with pytest.raises(AuthenticationError):
        tool.execute({})


def test_list_channels_tool_returns_channel():
    client = MagicMock()
    client.get_my_channel.return_value = ChannelSummary(
        channel_id="ch1",
        title="My Channel",
        description="",
        thumbnail_url=None,
        subscriber_count=10,
        video_count=2,
    )
    tool = ListMyChannelTool(client=client)
    result = tool.execute({"access_token": "tok"})
    assert result.success is True
    assert result.data["channel"]["channel_id"] == "ch1"


def test_my_uploads_tool_requires_access_token():
    tool = MyUploadsTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({})


def test_my_uploads_tool_returns_uploads_and_next_page_token():
    client = MagicMock()
    upload_item = {
        "playlist_item_id": "pi1",
        "playlist_id": "UUabc123",
        "video_id": "vid1",
        "title": "My Upload",
        "channel_title": "My Channel",
        "thumbnail_url": None,
        "position": 0,
    }
    client.get_my_uploads.return_value = ([_FakePlaylistItem(upload_item)], "next-token")
    tool = MyUploadsTool(client=client)

    result = tool.execute({"access_token": "tok", "max_results": 10})

    assert result.success is True
    assert result.data["uploads"][0]["video_id"] == "vid1"
    assert result.data["next_page_token"] == "next-token"
    client.get_my_uploads.assert_called_once_with(access_token="tok", max_results=10, page_token=None)


def test_my_uploads_tool_handles_empty_uploads():
    client = MagicMock()
    client.get_my_uploads.return_value = ([], None)
    tool = MyUploadsTool(client=client)

    result = tool.execute({"access_token": "tok"})

    assert result.success is True
    assert result.data["uploads"] == []
    assert result.data["next_page_token"] is None


def test_list_my_playlists_tool_requires_access_token():
    tool = ListMyPlaylistsTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({})


def test_list_my_playlists_tool_returns_playlists_and_next_page():
    client = MagicMock()
    playlist = {
        "playlist_id": "PL1",
        "title": "Watch Later",
        "description": "",
        "thumbnail_url": None,
        "privacy_status": "private",
        "item_count": 3,
    }
    client.list_my_playlists.return_value = ([_FakePlaylist(playlist)], "next-token")
    tool = ListMyPlaylistsTool(client=client)

    result = tool.execute({"access_token": "tok", "max_results": 10})

    assert result.success is True
    assert result.data["playlists"][0]["playlist_id"] == "PL1"
    assert result.data["next_page_token"] == "next-token"
    client.list_my_playlists.assert_called_once_with(access_token="tok", max_results=10, page_token=None)


def test_list_my_playlists_tool_handles_no_playlists():
    client = MagicMock()
    client.list_my_playlists.return_value = ([], None)
    tool = ListMyPlaylistsTool(client=client)

    result = tool.execute({"access_token": "tok"})

    assert result.success is True
    assert result.data["playlists"] == []
    assert result.data["next_page_token"] is None


def test_get_playlist_tool_requires_playlist_id():
    tool = GetPlaylistTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({})


def test_get_playlist_tool_works_without_access_token():
    """Reading one *known* playlist's public details/items has no
    account-specific meaning (unlike `youtube.playlist.list`, which is
    inherently "mine"), so `access_token` is optional here (mirrors
    `youtube.search`/`get_video`)."""
    client = MagicMock()
    playlist = {"playlist_id": "PL1", "title": "Public Mix", "description": "", "thumbnail_url": None, "privacy_status": "public", "item_count": 2}
    item = {
        "playlist_item_id": "pi1",
        "playlist_id": "PL1",
        "video_id": "vid1",
        "title": "First",
        "channel_title": "Chan",
        "thumbnail_url": None,
        "position": 0,
    }
    client.get_playlist.return_value = _FakePlaylist(playlist)
    client.list_playlist_items.return_value = ([_FakePlaylistItem(item)], None)
    tool = GetPlaylistTool(client=client)

    result = tool.execute({"playlist_id": "PL1"})

    assert result.success is True
    assert result.data["playlist"]["playlist_id"] == "PL1"
    assert result.data["items"][0]["video_id"] == "vid1"
    assert result.data["next_page_token"] is None
    client.get_playlist.assert_called_once_with(playlist_id="PL1", access_token=None)


def test_get_playlist_tool_handles_missing_playlist():
    client = MagicMock()
    client.get_playlist.return_value = None
    client.list_playlist_items.return_value = ([], None)
    tool = GetPlaylistTool(client=client)

    result = tool.execute({"playlist_id": "does-not-exist"})

    assert result.success is True
    assert result.data["playlist"] is None
    assert result.data["items"] == []


def test_create_playlist_tool_requires_title():
    tool = CreatePlaylistTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"access_token": "tok"})


def test_create_playlist_tool_requires_access_token():
    tool = CreatePlaylistTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"title": "My Mix"})


def test_create_playlist_tool_rejects_invalid_privacy_status():
    tool = CreatePlaylistTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"title": "My Mix", "access_token": "tok", "privacy_status": "not-real"})


def test_create_playlist_tool_creates_playlist_with_defaults():
    client = MagicMock()
    created = {
        "playlist_id": "PL1",
        "title": "My Mix",
        "description": "",
        "thumbnail_url": None,
        "privacy_status": "private",
        "item_count": 0,
    }
    client.create_playlist.return_value = _FakePlaylist(created)
    tool = CreatePlaylistTool(client=client)

    result = tool.execute({"title": "  My Mix  ", "access_token": "tok"})

    assert result.success is True
    assert result.data["playlist"]["playlist_id"] == "PL1"
    client.create_playlist.assert_called_once_with(
        title="My Mix", description="", privacy_status="private", access_token="tok"
    )


def test_update_playlist_tool_requires_playlist_id_and_title():
    tool = UpdatePlaylistTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"title": "Renamed", "access_token": "tok"})
    with pytest.raises(ValidationError):
        tool.execute({"playlist_id": "PL1", "access_token": "tok"})


def test_update_playlist_tool_requires_access_token():
    tool = UpdatePlaylistTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"playlist_id": "PL1", "title": "Renamed"})


def test_update_playlist_tool_rejects_invalid_privacy_status():
    tool = UpdatePlaylistTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"playlist_id": "PL1", "title": "Renamed", "access_token": "tok", "privacy_status": "not-real"})


def test_update_playlist_tool_updates_playlist():
    client = MagicMock()
    updated = {
        "playlist_id": "PL1",
        "title": "Renamed",
        "description": "new desc",
        "thumbnail_url": None,
        "privacy_status": "unlisted",
        "item_count": 3,
    }
    client.update_playlist.return_value = _FakePlaylist(updated)
    tool = UpdatePlaylistTool(client=client)

    result = tool.execute(
        {"playlist_id": "PL1", "title": "  Renamed  ", "privacy_status": "unlisted", "access_token": "tok"}
    )

    assert result.success is True
    assert result.data["playlist"]["title"] == "Renamed"
    client.update_playlist.assert_called_once_with(
        playlist_id="PL1", title="Renamed", description="", privacy_status="unlisted", access_token="tok"
    )


def test_delete_playlist_tool_requires_playlist_id():
    tool = DeletePlaylistTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"access_token": "tok"})


def test_delete_playlist_tool_requires_access_token():
    tool = DeletePlaylistTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"playlist_id": "PL1"})


def test_delete_playlist_tool_deletes_playlist():
    client = MagicMock()
    tool = DeletePlaylistTool(client=client)

    result = tool.execute({"playlist_id": "PL1", "access_token": "tok"})

    assert result.success is True
    assert result.data == {"deleted": True, "playlist_id": "PL1"}
    client.delete_playlist.assert_called_once_with(playlist_id="PL1", access_token="tok")


def test_add_playlist_item_tool_requires_playlist_id_and_video_id():
    tool = AddPlaylistItemTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"video_id": "vid1", "access_token": "tok"})
    with pytest.raises(ValidationError):
        tool.execute({"playlist_id": "PL1", "access_token": "tok"})


def test_add_playlist_item_tool_requires_access_token():
    tool = AddPlaylistItemTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"playlist_id": "PL1", "video_id": "vid1"})


def test_add_playlist_item_tool_adds_video():
    client = MagicMock()
    added = {
        "playlist_item_id": "pi1",
        "playlist_id": "PL1",
        "video_id": "vid1",
        "title": "First",
        "channel_title": "Chan",
        "thumbnail_url": None,
        "position": None,
    }
    client.add_playlist_item.return_value = _FakePlaylistItem(added)
    tool = AddPlaylistItemTool(client=client)

    result = tool.execute({"playlist_id": "PL1", "video_id": "vid1", "access_token": "tok"})

    assert result.success is True
    assert result.data["item"]["playlist_item_id"] == "pi1"
    client.add_playlist_item.assert_called_once_with(playlist_id="PL1", video_id="vid1", position=None, access_token="tok")


def test_remove_playlist_item_tool_requires_playlist_item_id():
    tool = RemovePlaylistItemTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"access_token": "tok"})


def test_remove_playlist_item_tool_requires_access_token():
    tool = RemovePlaylistItemTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"playlist_item_id": "pi1"})


def test_remove_playlist_item_tool_removes_item():
    client = MagicMock()
    tool = RemovePlaylistItemTool(client=client)

    result = tool.execute({"playlist_item_id": "pi1", "access_token": "tok"})

    assert result.success is True
    assert result.data == {"removed": True, "playlist_item_id": "pi1"}
    client.remove_playlist_item.assert_called_once_with(playlist_item_id="pi1", access_token="tok")


def test_reorder_playlist_item_tool_requires_all_fields():
    tool = ReorderPlaylistItemTool(client=MagicMock())
    base = {"playlist_item_id": "pi1", "playlist_id": "PL1", "video_id": "vid1", "position": 0, "access_token": "tok"}
    for missing in ("playlist_item_id", "playlist_id", "video_id", "position"):
        args = dict(base)
        del args[missing]
        with pytest.raises(ValidationError):
            tool.execute(args)


def test_reorder_playlist_item_tool_reorders_item():
    client = MagicMock()
    moved = {
        "playlist_item_id": "pi1",
        "playlist_id": "PL1",
        "video_id": "vid1",
        "title": "First",
        "channel_title": "Chan",
        "thumbnail_url": None,
        "position": 0,
    }
    client.reorder_playlist_item.return_value = _FakePlaylistItem(moved)
    tool = ReorderPlaylistItemTool(client=client)

    result = tool.execute(
        {"playlist_item_id": "pi1", "playlist_id": "PL1", "video_id": "vid1", "position": 0, "access_token": "tok"}
    )

    assert result.success is True
    assert result.data["item"]["position"] == 0
    client.reorder_playlist_item.assert_called_once_with(
        playlist_item_id="pi1", playlist_id="PL1", video_id="vid1", position=0, access_token="tok"
    )


class _FakePlaylist:
    """Minimal stand-in for `PlaylistSummary` exposing only `to_dict()`."""

    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return dict(self._data)


class _FakePlaylistItem:
    """Minimal stand-in for `PlaylistItem` exposing only `to_dict()`,
    since `MyUploadsTool.execute` only ever calls that method on each
    element `get_my_uploads` returns."""

    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return dict(self._data)
