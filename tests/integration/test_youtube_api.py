from app.integrations.youtube_api import ChannelSummary, PlaylistItem, PlaylistSummary, VideoDetails, VideoSummary

SUMMARY = VideoSummary(
    video_id="vid1",
    title="Python Tutorial",
    channel_title="Code Channel",
    description="desc",
    thumbnail_url=None,
    published_at=None,
    url="https://www.youtube.com/watch?v=vid1",
)


def test_search_endpoint_requires_query(client):
    resp = client.get("/api/v1/youtube/search")
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_search_endpoint_without_api_key_returns_classified_error(client):
    # No YOUTUBE_API_KEY set and no Google account connected in tests.
    resp = client.get("/api/v1/youtube/search?q=python")
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_search_endpoint_returns_results(client, monkeypatch):
    def fake_search(self, *, query, max_results, order, access_token):
        return [SUMMARY]

    monkeypatch.setattr("app.integrations.youtube_api.YouTubeApiClient.search", fake_search)

    resp = client.get("/api/v1/youtube/search?q=python+tutorial&max_results=5")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["query"] == "python tutorial"
    assert body["results"][0]["video_id"] == "vid1"
    assert body["request_id"] is not None


def test_get_video_endpoint(client, monkeypatch):
    details = VideoDetails(**SUMMARY.__dict__, duration_iso8601="PT5M", view_count=1, like_count=1)

    def fake_get_videos(self, *, video_ids, access_token):
        return [details]

    monkeypatch.setattr("app.integrations.youtube_api.YouTubeApiClient.get_videos", fake_get_videos)

    resp = client.get("/api/v1/youtube/videos/vid1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["results"][0]["video_id"] == "vid1"


def test_my_channel_endpoint_requires_connected_account(client):
    resp = client.get("/api/v1/youtube/channels/mine")
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "AUTHENTICATION_ERROR"


def test_my_channel_endpoint_returns_channel(client, monkeypatch):
    channel = ChannelSummary(
        channel_id="ch1",
        title="My Channel",
        description="About my channel",
        thumbnail_url="https://img/high.jpg",
        subscriber_count=1234,
        video_count=56,
    )

    monkeypatch.setattr("app.api.v1.youtube.resolve_access_token", lambda: "fake-token")
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.get_my_channel",
        lambda self, *, access_token: channel,
    )

    resp = client.get("/api/v1/youtube/channels/mine")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["channel"]["channel_id"] == "ch1"
    assert body["channel"]["subscriber_count"] == 1234
    assert body["request_id"] is not None


def test_my_uploads_endpoint_requires_connected_account(client):
    resp = client.get("/api/v1/youtube/channels/mine/uploads")
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "AUTHENTICATION_ERROR"


def test_my_uploads_endpoint_returns_uploads(client, monkeypatch):
    upload = PlaylistItem(
        playlist_item_id="pi1",
        playlist_id="UUabc123",
        video_id="vid1",
        title="My Upload",
        channel_title="My Channel",
        thumbnail_url=None,
        position=0,
    )

    monkeypatch.setattr("app.api.v1.youtube.require_access_token", lambda message: "fake-token")
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.get_my_uploads",
        lambda self, *, access_token, max_results, page_token: ([upload], "next-token"),
    )

    resp = client.get("/api/v1/youtube/channels/mine/uploads")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["uploads"][0]["video_id"] == "vid1"
    assert body["next_page_token"] == "next-token"
    assert len(body["cards"]) == 1
    assert body["cards"][0]["type"] == "youtube_playlist_item"


def test_list_playlists_endpoint_requires_connected_account(client):
    resp = client.get("/api/v1/youtube/playlists")
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "AUTHENTICATION_ERROR"


def test_list_playlists_endpoint_returns_playlists(client, monkeypatch):
    playlist = PlaylistSummary(
        playlist_id="PL1",
        title="Watch Later",
        description="",
        thumbnail_url=None,
        privacy_status="private",
        item_count=3,
    )

    monkeypatch.setattr("app.api.v1.youtube.require_access_token", lambda message: "fake-token")
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.list_my_playlists",
        lambda self, *, access_token, max_results, page_token: ([playlist], "next-token"),
    )

    resp = client.get("/api/v1/youtube/playlists")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["playlists"][0]["playlist_id"] == "PL1"
    assert body["next_page_token"] == "next-token"
    assert len(body["cards"]) == 1
    assert body["cards"][0]["type"] == "youtube_playlist"


def test_list_playlists_endpoint_returns_empty_list(client, monkeypatch):
    monkeypatch.setattr("app.api.v1.youtube.require_access_token", lambda message: "fake-token")
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.list_my_playlists",
        lambda self, *, access_token, max_results, page_token: ([], None),
    )

    resp = client.get("/api/v1/youtube/playlists")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["playlists"] == []
    assert body["cards"] == []


def test_create_playlist_endpoint_requires_connected_account(client):
    resp = client.post("/api/v1/youtube/playlists", json={"title": "My Mix"})
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "AUTHENTICATION_ERROR"


def test_create_playlist_endpoint_requires_title(client, monkeypatch):
    monkeypatch.setattr("app.api.v1.youtube.require_access_token", lambda message: "fake-token")

    resp = client.post("/api/v1/youtube/playlists", json={})
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_create_playlist_endpoint_returns_created_playlist(client, monkeypatch):
    playlist = PlaylistSummary(
        playlist_id="PL1",
        title="My Mix",
        description="",
        thumbnail_url=None,
        privacy_status="private",
        item_count=0,
    )

    monkeypatch.setattr("app.api.v1.youtube.require_access_token", lambda message: "fake-token")
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.create_playlist",
        lambda self, *, title, description, privacy_status, access_token: playlist,
    )

    resp = client.post("/api/v1/youtube/playlists", json={"title": "My Mix"})
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["success"] is True
    assert body["playlist"]["playlist_id"] == "PL1"


def test_update_playlist_endpoint_requires_connected_account(client):
    resp = client.put("/api/v1/youtube/playlists/PL1", json={"title": "Renamed"})
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "AUTHENTICATION_ERROR"


def test_update_playlist_endpoint_requires_title(client, monkeypatch):
    monkeypatch.setattr("app.api.v1.youtube.require_access_token", lambda message: "fake-token")

    resp = client.put("/api/v1/youtube/playlists/PL1", json={})
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_update_playlist_endpoint_returns_updated_playlist(client, monkeypatch):
    playlist = PlaylistSummary(
        playlist_id="PL1",
        title="Renamed",
        description="",
        thumbnail_url=None,
        privacy_status="unlisted",
        item_count=3,
    )

    monkeypatch.setattr("app.api.v1.youtube.require_access_token", lambda message: "fake-token")
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.update_playlist",
        lambda self, *, playlist_id, title, description, privacy_status, access_token: playlist,
    )

    resp = client.put("/api/v1/youtube/playlists/PL1", json={"title": "Renamed", "privacy_status": "unlisted"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["playlist"]["title"] == "Renamed"


def test_delete_playlist_endpoint_requires_connected_account(client):
    resp = client.delete("/api/v1/youtube/playlists/PL1")
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "AUTHENTICATION_ERROR"


def test_delete_playlist_endpoint_returns_deleted_true(client, monkeypatch):
    monkeypatch.setattr("app.api.v1.youtube.require_access_token", lambda message: "fake-token")
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.delete_playlist",
        lambda self, *, playlist_id, access_token: None,
    )

    resp = client.delete("/api/v1/youtube/playlists/PL1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["deleted"] is True


def test_add_playlist_item_endpoint_requires_connected_account(client):
    resp = client.post("/api/v1/youtube/playlists/PL1/items", json={"video_id": "vid1"})
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "AUTHENTICATION_ERROR"


def test_add_playlist_item_endpoint_requires_video_id(client, monkeypatch):
    monkeypatch.setattr("app.api.v1.youtube.require_access_token", lambda message: "fake-token")

    resp = client.post("/api/v1/youtube/playlists/PL1/items", json={})
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_add_playlist_item_endpoint_returns_created_item(client, monkeypatch):
    item = PlaylistItem(
        playlist_item_id="pi1",
        playlist_id="PL1",
        video_id="vid1",
        title="First",
        channel_title="Chan",
        thumbnail_url=None,
        position=None,
    )

    monkeypatch.setattr("app.api.v1.youtube.require_access_token", lambda message: "fake-token")
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.add_playlist_item",
        lambda self, *, playlist_id, video_id, position, access_token: item,
    )

    resp = client.post("/api/v1/youtube/playlists/PL1/items", json={"video_id": "vid1"})
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["success"] is True
    assert body["item"]["playlist_item_id"] == "pi1"


def test_remove_playlist_item_endpoint_requires_connected_account(client):
    resp = client.delete("/api/v1/youtube/playlists/PL1/items/pi1")
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "AUTHENTICATION_ERROR"


def test_remove_playlist_item_endpoint_returns_removed_true(client, monkeypatch):
    monkeypatch.setattr("app.api.v1.youtube.require_access_token", lambda message: "fake-token")
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.remove_playlist_item",
        lambda self, *, playlist_item_id, access_token: None,
    )

    resp = client.delete("/api/v1/youtube/playlists/PL1/items/pi1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["removed"] is True


def test_reorder_playlist_item_endpoint_requires_connected_account(client):
    resp = client.post("/api/v1/youtube/playlists/PL1/items/pi1/reorder", json={"video_id": "vid1", "position": 0})
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "AUTHENTICATION_ERROR"


def test_reorder_playlist_item_endpoint_requires_position(client, monkeypatch):
    monkeypatch.setattr("app.api.v1.youtube.require_access_token", lambda message: "fake-token")

    resp = client.post("/api/v1/youtube/playlists/PL1/items/pi1/reorder", json={"video_id": "vid1"})
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_reorder_playlist_item_endpoint_returns_moved_item(client, monkeypatch):
    item = PlaylistItem(
        playlist_item_id="pi1",
        playlist_id="PL1",
        video_id="vid1",
        title="First",
        channel_title="Chan",
        thumbnail_url=None,
        position=0,
    )

    monkeypatch.setattr("app.api.v1.youtube.require_access_token", lambda message: "fake-token")
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.reorder_playlist_item",
        lambda self, *, playlist_item_id, playlist_id, video_id, position, access_token: item,
    )

    resp = client.post("/api/v1/youtube/playlists/PL1/items/pi1/reorder", json={"video_id": "vid1", "position": 0})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["item"]["position"] == 0


def test_get_playlist_endpoint_returns_playlist_and_items(client, monkeypatch):
    playlist = PlaylistSummary(
        playlist_id="PL1",
        title="Watch Later",
        description="",
        thumbnail_url=None,
        privacy_status="private",
        item_count=1,
    )
    item = PlaylistItem(
        playlist_item_id="pi1",
        playlist_id="PL1",
        video_id="vid1",
        title="First",
        channel_title="Chan",
        thumbnail_url=None,
        position=0,
    )

    monkeypatch.setattr("app.api.v1.youtube.resolve_access_token", lambda: None)
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.get_playlist",
        lambda self, *, playlist_id, access_token: playlist,
    )
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.list_playlist_items",
        lambda self, *, playlist_id, max_results, page_token, access_token: ([item], None),
    )

    resp = client.get("/api/v1/youtube/playlists/PL1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["playlist"]["playlist_id"] == "PL1"
    assert body["items"][0]["video_id"] == "vid1"
    assert len(body["cards"]) == 1
    assert body["cards"][0]["type"] == "youtube_playlist_item"


def test_get_playlist_endpoint_handles_missing_playlist(client, monkeypatch):
    monkeypatch.setattr("app.api.v1.youtube.resolve_access_token", lambda: None)
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.get_playlist",
        lambda self, *, playlist_id, access_token: None,
    )
    monkeypatch.setattr(
        "app.integrations.youtube_api.YouTubeApiClient.list_playlist_items",
        lambda self, *, playlist_id, max_results, page_token, access_token: ([], None),
    )

    resp = client.get("/api/v1/youtube/playlists/does-not-exist")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["playlist"] is None
    assert body["items"] == []
