from unittest.mock import MagicMock, patch

import pytest

from app.core.errors import AuthenticationError, NetworkError, RateLimitError, TimeoutErrorNova, ValidationError
from app.integrations.youtube_api import YouTubeApiClient

SEARCH_BODY = {
    "items": [
        {
            "id": {"videoId": "vid1"},
            "snippet": {
                "title": "Python Tutorial",
                "channelTitle": "Code Channel",
                "description": "Learn Python",
                "thumbnails": {"high": {"url": "https://img/high.jpg"}},
                "publishedAt": "2024-01-01T00:00:00Z",
            },
        }
    ]
}

VIDEOS_BODY = {
    "items": [
        {
            "id": "vid1",
            "snippet": {
                "title": "Python Tutorial",
                "channelTitle": "Code Channel",
                "description": "Learn Python",
                "thumbnails": {},
                "publishedAt": "2024-01-01T00:00:00Z",
            },
            "contentDetails": {"duration": "PT10M"},
            "statistics": {"viewCount": "1000", "likeCount": "50"},
        }
    ]
}


def _mock_response(status_code=200, json_body=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body or {}
    resp.text = text
    return resp


def test_search_requires_api_key_or_access_token():
    client = YouTubeApiClient(api_key="")
    with pytest.raises(ValidationError):
        client.search(query="python", max_results=5, order="relevance", access_token=None)


def test_search_uses_api_key_and_parses_results():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body=SEARCH_BODY)) as mock_get:
        results = client.search(query="python", max_results=5, order="relevance", access_token=None)

    assert len(results) == 1
    assert results[0].video_id == "vid1"
    assert results[0].title == "Python Tutorial"
    assert results[0].url == "https://www.youtube.com/watch?v=vid1"
    called_params = mock_get.call_args.kwargs["params"]
    assert called_params["key"] == "test-key"


def test_search_prefers_access_token_over_api_key():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body=SEARCH_BODY)) as mock_get:
        client.search(query="python", max_results=5, order="relevance", access_token="oauth-token")

    headers = mock_get.call_args.kwargs["headers"]
    params = mock_get.call_args.kwargs["params"]
    assert headers["Authorization"] == "Bearer oauth-token"
    assert "key" not in params


def test_get_videos_parses_details():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body=VIDEOS_BODY)):
        results = client.get_videos(video_ids=["vid1"], access_token=None)

    assert len(results) == 1
    assert results[0].duration_iso8601 == "PT10M"
    assert results[0].view_count == 1000
    assert results[0].like_count == 50


def test_get_videos_empty_list_short_circuits():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get") as mock_get:
        results = client.get_videos(video_ids=[], access_token=None)
    assert results == []
    mock_get.assert_not_called()


def test_get_my_channel_requires_access_token():
    client = YouTubeApiClient(api_key="test-key")
    with pytest.raises(AuthenticationError):
        client.get_my_channel(access_token="")


def test_get_my_channel_returns_none_when_no_channel():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body={"items": []})):
        channel = client.get_my_channel(access_token="tok")
    assert channel is None


def test_get_my_channel_returns_parsed_channel():
    client = YouTubeApiClient(api_key="test-key")
    body = {
        "items": [
            {
                "id": "ch1",
                "snippet": {
                    "title": "My Channel",
                    "description": "About my channel",
                    "thumbnails": {"high": {"url": "https://img/high.jpg"}},
                },
                "statistics": {"subscriberCount": "1234", "videoCount": "56"},
                "contentDetails": {"relatedPlaylists": {"uploads": "UUabc123"}},
            }
        ]
    }
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body=body)) as mock_get:
        channel = client.get_my_channel(access_token="tok")

    assert channel.channel_id == "ch1"
    assert channel.title == "My Channel"
    assert channel.description == "About my channel"
    assert channel.thumbnail_url == "https://img/high.jpg"
    assert channel.subscriber_count == 1234
    assert channel.video_count == 56
    assert channel.uploads_playlist_id == "UUabc123"
    # `mine=true` is what makes this "my channel" rather than a public lookup.
    assert mock_get.call_args.kwargs["params"]["mine"] == "true"


def test_get_my_uploads_returns_empty_when_no_channel():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body={"items": []})) as mock_get:
        uploads, next_token = client.get_my_uploads(access_token="tok")

    assert uploads == []
    assert next_token is None
    # Only the channel lookup should fire - no playlistItems call once
    # there's no uploads playlist to look up.
    assert mock_get.call_count == 1


def test_get_my_uploads_returns_empty_when_channel_has_no_uploads_playlist():
    client = YouTubeApiClient(api_key="test-key")
    body = {
        "items": [
            {
                "id": "ch1",
                "snippet": {"title": "My Channel", "description": "", "thumbnails": {}},
                "statistics": {},
                "contentDetails": {},  # no relatedPlaylists.uploads at all
            }
        ]
    }
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body=body)) as mock_get:
        uploads, next_token = client.get_my_uploads(access_token="tok")

    assert uploads == []
    assert next_token is None
    assert mock_get.call_count == 1


def test_get_my_uploads_fetches_uploads_playlist_items():
    client = YouTubeApiClient(api_key="test-key")
    channel_body = {
        "items": [
            {
                "id": "ch1",
                "snippet": {"title": "My Channel", "description": "", "thumbnails": {}},
                "statistics": {"subscriberCount": "10", "videoCount": "1"},
                "contentDetails": {"relatedPlaylists": {"uploads": "UUabc123"}},
            }
        ]
    }
    items_body = {
        "items": [
            {
                "id": "pi1",
                "snippet": {
                    "playlistId": "UUabc123",
                    "resourceId": {"videoId": "vid1"},
                    "title": "My Upload",
                    "channelTitle": "My Channel",
                    "thumbnails": {},
                    "position": 0,
                },
            }
        ],
        "nextPageToken": "next-token",
    }
    with patch(
        "app.integrations.youtube_api.requests.get",
        side_effect=[_mock_response(json_body=channel_body), _mock_response(json_body=items_body)],
    ) as mock_get:
        uploads, next_token = client.get_my_uploads(access_token="tok", max_results=10)

    assert len(uploads) == 1
    assert uploads[0].video_id == "vid1"
    assert uploads[0].playlist_id == "UUabc123"
    assert next_token == "next-token"
    assert mock_get.call_count == 2
    # Second call is the playlistItems lookup, scoped to the channel's
    # actual uploads playlist (not a hardcoded/guessed ID).
    second_call_params = mock_get.call_args_list[1].kwargs["params"]
    assert second_call_params["playlistId"] == "UUabc123"


def test_401_raises_authentication_error():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(status_code=401)):
        with pytest.raises(AuthenticationError):
            client.search(query="x", max_results=1, order="relevance", access_token=None)


def test_403_quota_raises_rate_limit_error():
    client = YouTubeApiClient(api_key="test-key")
    body = {"error": {"message": "Quota exceeded for quota metric"}}
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(status_code=403, json_body=body)):
        with pytest.raises(RateLimitError):
            client.search(query="x", max_results=1, order="relevance", access_token=None)


def test_403_non_quota_raises_authentication_error():
    client = YouTubeApiClient(api_key="test-key")
    body = {"error": {"message": "Access not configured"}}
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(status_code=403, json_body=body)):
        with pytest.raises(AuthenticationError):
            client.search(query="x", max_results=1, order="relevance", access_token=None)


def test_timeout_raises_timeout_error():
    import requests as requests_module

    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", side_effect=requests_module.Timeout("boom")):
        with pytest.raises(TimeoutErrorNova):
            client.search(query="x", max_results=1, order="relevance", access_token=None)


def test_connection_error_raises_network_error():
    import requests as requests_module

    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", side_effect=requests_module.ConnectionError("boom")):
        with pytest.raises(NetworkError):
            client.search(query="x", max_results=1, order="relevance", access_token=None)


def test_list_my_playlists_requires_access_token():
    client = YouTubeApiClient(api_key="test-key")
    with pytest.raises(AuthenticationError):
        client.list_my_playlists(access_token="")


def test_list_my_playlists_returns_playlists_and_scopes_to_mine():
    client = YouTubeApiClient(api_key="test-key")
    body = {
        "items": [
            {
                "id": "PL1",
                "snippet": {"title": "Watch Later", "description": "", "thumbnails": {}},
                "status": {"privacyStatus": "private"},
                "contentDetails": {"itemCount": 3},
            }
        ],
        "nextPageToken": "next-token",
    }
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body=body)) as mock_get:
        playlists, next_token = client.list_my_playlists(access_token="tok", max_results=10)

    assert len(playlists) == 1
    assert playlists[0].playlist_id == "PL1"
    assert playlists[0].item_count == 3
    assert next_token == "next-token"
    # `mine=true` is what scopes this to the connected account's own
    # playlists, not an arbitrary/public list.
    assert mock_get.call_args.kwargs["params"]["mine"] == "true"


def test_list_my_playlists_handles_no_playlists():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body={"items": []})):
        playlists, next_token = client.list_my_playlists(access_token="tok")

    assert playlists == []
    assert next_token is None


def test_get_playlist_returns_none_when_not_found():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body={"items": []})):
        playlist = client.get_playlist(playlist_id="does-not-exist")

    assert playlist is None


def test_create_playlist_requires_access_token():
    client = YouTubeApiClient(api_key="test-key")
    with pytest.raises(AuthenticationError):
        client.create_playlist(title="My Mix", access_token="")


def test_create_playlist_posts_snippet_and_status_and_parses_result():
    client = YouTubeApiClient(api_key="test-key")
    body = {
        "id": "PL1",
        "snippet": {"title": "My Mix", "description": "desc", "thumbnails": {}},
        "status": {"privacyStatus": "private"},
    }
    with patch("app.integrations.youtube_api.requests.request", return_value=_mock_response(201, json_body=body)) as mock_req:
        playlist = client.create_playlist(title="My Mix", description="desc", access_token="tok")

    assert playlist.playlist_id == "PL1"
    assert playlist.title == "My Mix"
    assert playlist.privacy_status == "private"
    assert mock_req.call_args.args[0] == "post"
    assert mock_req.call_args.kwargs["json"]["snippet"] == {"title": "My Mix", "description": "desc"}
    assert mock_req.call_args.kwargs["json"]["status"] == {"privacyStatus": "private"}
    assert mock_req.call_args.kwargs["headers"]["Authorization"] == "Bearer tok"


def test_create_playlist_defaults_to_private_and_empty_description():
    client = YouTubeApiClient(api_key="test-key")
    body = {"id": "PL2", "snippet": {"title": "No Frills", "description": "", "thumbnails": {}}, "status": {"privacyStatus": "private"}}
    with patch("app.integrations.youtube_api.requests.request", return_value=_mock_response(201, json_body=body)) as mock_req:
        client.create_playlist(title="No Frills", access_token="tok")

    assert mock_req.call_args.kwargs["json"]["snippet"]["description"] == ""
    assert mock_req.call_args.kwargs["json"]["status"]["privacyStatus"] == "private"


def test_update_playlist_requires_access_token():
    client = YouTubeApiClient(api_key="test-key")
    with pytest.raises(AuthenticationError):
        client.update_playlist(playlist_id="PL1", title="Renamed", access_token="")


def test_update_playlist_puts_id_snippet_and_status_and_parses_result():
    client = YouTubeApiClient(api_key="test-key")
    body = {
        "id": "PL1",
        "snippet": {"title": "Renamed", "description": "new desc", "thumbnails": {}},
        "status": {"privacyStatus": "unlisted"},
    }
    with patch("app.integrations.youtube_api.requests.request", return_value=_mock_response(200, json_body=body)) as mock_req:
        playlist = client.update_playlist(
            playlist_id="PL1", title="Renamed", description="new desc", privacy_status="unlisted", access_token="tok"
        )

    assert playlist.playlist_id == "PL1"
    assert playlist.title == "Renamed"
    assert playlist.privacy_status == "unlisted"
    assert mock_req.call_args.args[0] == "put"
    assert mock_req.call_args.kwargs["json"]["id"] == "PL1"
    assert mock_req.call_args.kwargs["json"]["snippet"] == {"title": "Renamed", "description": "new desc"}
    assert mock_req.call_args.kwargs["json"]["status"] == {"privacyStatus": "unlisted"}


def test_delete_playlist_requires_access_token():
    client = YouTubeApiClient(api_key="test-key")
    with pytest.raises(AuthenticationError):
        client.delete_playlist(playlist_id="PL1", access_token="")


def test_delete_playlist_sends_delete_with_id_param():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.request", return_value=_mock_response(204)) as mock_req:
        result = client.delete_playlist(playlist_id="PL1", access_token="tok")

    assert result is None
    assert mock_req.call_args.args[0] == "delete"
    assert mock_req.call_args.kwargs["params"]["id"] == "PL1"
    assert mock_req.call_args.kwargs["headers"]["Authorization"] == "Bearer tok"


def test_add_playlist_item_requires_access_token():
    client = YouTubeApiClient(api_key="test-key")
    with pytest.raises(AuthenticationError):
        client.add_playlist_item(playlist_id="PL1", video_id="vid1", access_token="")


def test_add_playlist_item_posts_snippet_with_optional_position():
    client = YouTubeApiClient(api_key="test-key")
    body = {
        "id": "pi1",
        "snippet": {
            "playlistId": "PL1",
            "position": 2,
            "resourceId": {"videoId": "vid1"},
            "title": "First",
            "channelTitle": "Chan",
            "thumbnails": {},
        },
    }
    with patch("app.integrations.youtube_api.requests.request", return_value=_mock_response(201, json_body=body)) as mock_req:
        item = client.add_playlist_item(playlist_id="PL1", video_id="vid1", position=2, access_token="tok")

    assert item.playlist_item_id == "pi1"
    assert item.video_id == "vid1"
    assert mock_req.call_args.args[0] == "post"
    sent_snippet = mock_req.call_args.kwargs["json"]["snippet"]
    assert sent_snippet["playlistId"] == "PL1"
    assert sent_snippet["resourceId"] == {"kind": "youtube#video", "videoId": "vid1"}
    assert sent_snippet["position"] == 2


def test_add_playlist_item_omits_position_when_not_given():
    client = YouTubeApiClient(api_key="test-key")
    body = {"id": "pi1", "snippet": {"playlistId": "PL1", "resourceId": {"videoId": "vid1"}, "thumbnails": {}}}
    with patch("app.integrations.youtube_api.requests.request", return_value=_mock_response(201, json_body=body)) as mock_req:
        client.add_playlist_item(playlist_id="PL1", video_id="vid1", access_token="tok")

    assert "position" not in mock_req.call_args.kwargs["json"]["snippet"]


def test_remove_playlist_item_requires_access_token():
    client = YouTubeApiClient(api_key="test-key")
    with pytest.raises(AuthenticationError):
        client.remove_playlist_item(playlist_item_id="pi1", access_token="")


def test_remove_playlist_item_sends_delete_with_id_param():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.request", return_value=_mock_response(204)) as mock_req:
        result = client.remove_playlist_item(playlist_item_id="pi1", access_token="tok")

    assert result is None
    assert mock_req.call_args.args[0] == "delete"
    assert mock_req.call_args.kwargs["params"]["id"] == "pi1"


def test_reorder_playlist_item_requires_access_token():
    client = YouTubeApiClient(api_key="test-key")
    with pytest.raises(AuthenticationError):
        client.reorder_playlist_item(playlist_item_id="pi1", playlist_id="PL1", video_id="vid1", position=0, access_token="")


def test_reorder_playlist_item_puts_id_and_new_position():
    client = YouTubeApiClient(api_key="test-key")
    body = {
        "id": "pi1",
        "snippet": {"playlistId": "PL1", "position": 0, "resourceId": {"videoId": "vid1"}, "thumbnails": {}},
    }
    with patch("app.integrations.youtube_api.requests.request", return_value=_mock_response(200, json_body=body)) as mock_req:
        item = client.reorder_playlist_item(playlist_item_id="pi1", playlist_id="PL1", video_id="vid1", position=0, access_token="tok")

    assert item.position == 0
    assert mock_req.call_args.args[0] == "put"
    sent = mock_req.call_args.kwargs["json"]
    assert sent["id"] == "pi1"
    assert sent["snippet"]["position"] == 0
    assert sent["snippet"]["playlistId"] == "PL1"


def test_get_playlist_works_with_api_key_only():
    """Unlike `list_my_playlists` (inherently 'mine'), looking up one known
    public playlist by ID needs no OAuth token - same public/API-key path
    as `search()`/`get_videos()`."""
    client = YouTubeApiClient(api_key="test-key")
    body = {
        "items": [
            {
                "id": "PL1",
                "snippet": {"title": "Public Mix", "description": "", "thumbnails": {}},
                "status": {"privacyStatus": "public"},
                "contentDetails": {"itemCount": 5},
            }
        ]
    }
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body=body)) as mock_get:
        playlist = client.get_playlist(playlist_id="PL1", access_token=None)

    assert playlist.playlist_id == "PL1"
    assert playlist.title == "Public Mix"
    assert mock_get.call_args.kwargs["params"]["id"] == "PL1"
