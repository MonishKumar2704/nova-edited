"""Additional tests for YouTube *search* (Task 37: verify YouTube search).

`youtube.search` already had solid coverage before this task -
`test_youtube_api_client.py` (api-key vs OAuth-token precedence, full
401/403-quota/403-other/timeout/network error matrix) and
`test_youtube_tools.py` (query/order validation, basic caching) - unlike
Gmail's initial state (Tasks 25-36). This file closes the specific gaps
found while re-verifying end to end:

- `max_results` is documented as accepting 1-25 but the clamp
  (`max(1, min(max_results, 25))`) itself was never exercised - only
  values already inside the valid range were ever passed in a test.
- A malformed search-result item (no `videoId`, e.g. YouTube occasionally
  returns a channel/playlist result mixed into `type=video` search
  results) is silently skipped by `search()`, but nothing proved that.
- An empty `items` list (no results for the query) was never exercised.
- `_best_thumbnail`'s high > medium > default preference, and its `None`
  fallback when a video has no thumbnails at all, were never exercised -
  this runs on every search result, so a regression here would silently
  degrade every result card.
- `order` was asserted to reach the request in `test_youtube_tools.py`
  only indirectly (via the tool passing it through); nothing checked the
  literal `order`/`maxResults` values landing in the actual HTTP request
  parameters sent to the YouTube Data API.
- The tool-level cache was proven to cache identical calls, but not that
  a call with an access token is kept separate from one without (an
  authenticated user's search shouldn't be transparently served a
  cache entry populated by an anonymous API-key search, or vice versa -
  results could legitimately differ, e.g. account-specific quota/region).
"""

from unittest.mock import MagicMock, patch

from app.core.cache import TTLCache
from app.integrations.youtube_api import YouTubeApiClient
from app.tools.youtube.search_videos import SearchVideosTool


def _mock_response(status_code=200, json_body=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body or {}
    resp.text = text
    return resp


# -- max_results clamping (client) -------------------------------------------


def test_search_clamps_max_results_above_25_down_to_25():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body={"items": []})) as mock_get:
        client.search(query="python", max_results=100, order="relevance", access_token=None)
    assert mock_get.call_args.kwargs["params"]["maxResults"] == 25


def test_search_clamps_max_results_below_1_up_to_1():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body={"items": []})) as mock_get:
        client.search(query="python", max_results=0, order="relevance", access_token=None)
    assert mock_get.call_args.kwargs["params"]["maxResults"] == 1


def test_search_sends_order_and_type_video_in_request_params():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body={"items": []})) as mock_get:
        client.search(query="python", max_results=5, order="viewCount", access_token=None)
    params = mock_get.call_args.kwargs["params"]
    assert params["order"] == "viewCount"
    assert params["type"] == "video"
    assert params["q"] == "python"


# -- malformed / empty result handling ---------------------------------------


def test_search_skips_items_with_no_video_id():
    client = YouTubeApiClient(api_key="test-key")
    body = {
        "items": [
            {"id": {"kind": "youtube#channel", "channelId": "c1"}, "snippet": {"title": "Not a video"}},
            {"id": {"videoId": "vid1"}, "snippet": {"title": "Real video"}},
        ]
    }
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body=body)):
        results = client.search(query="python", max_results=5, order="relevance", access_token=None)
    assert len(results) == 1
    assert results[0].video_id == "vid1"


def test_search_returns_empty_list_for_no_results():
    client = YouTubeApiClient(api_key="test-key")
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body={"items": []})):
        results = client.search(query="asdkjhqwelkjhasd", max_results=5, order="relevance", access_token=None)
    assert results == []


# -- thumbnail preference -----------------------------------------------------


def test_search_prefers_high_over_medium_over_default_thumbnail():
    client = YouTubeApiClient(api_key="test-key")
    body = {
        "items": [
            {
                "id": {"videoId": "vid1"},
                "snippet": {
                    "title": "T",
                    "thumbnails": {
                        "default": {"url": "https://img/default.jpg"},
                        "medium": {"url": "https://img/medium.jpg"},
                        "high": {"url": "https://img/high.jpg"},
                    },
                },
            }
        ]
    }
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body=body)):
        results = client.search(query="python", max_results=1, order="relevance", access_token=None)
    assert results[0].thumbnail_url == "https://img/high.jpg"


def test_search_falls_back_to_medium_then_default_thumbnail():
    client = YouTubeApiClient(api_key="test-key")
    body = {
        "items": [
            {"id": {"videoId": "vid1"}, "snippet": {"title": "T", "thumbnails": {"medium": {"url": "https://img/medium.jpg"}}}},
            {"id": {"videoId": "vid2"}, "snippet": {"title": "T2", "thumbnails": {"default": {"url": "https://img/default.jpg"}}}},
        ]
    }
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body=body)):
        results = client.search(query="python", max_results=2, order="relevance", access_token=None)
    assert results[0].thumbnail_url == "https://img/medium.jpg"
    assert results[1].thumbnail_url == "https://img/default.jpg"


def test_search_thumbnail_is_none_when_video_has_no_thumbnails():
    client = YouTubeApiClient(api_key="test-key")
    body = {"items": [{"id": {"videoId": "vid1"}, "snippet": {"title": "T"}}]}
    with patch("app.integrations.youtube_api.requests.get", return_value=_mock_response(json_body=body)):
        results = client.search(query="python", max_results=1, order="relevance", access_token=None)
    assert results[0].thumbnail_url is None


# -- tool-level cache separation by auth state -------------------------------


def test_search_tool_keeps_separate_cache_entries_for_authed_vs_anonymous():
    client = MagicMock()
    client.search.side_effect = [
        [MagicMock(to_dict=lambda: {"video_id": "public-result"})],
        [MagicMock(to_dict=lambda: {"video_id": "authed-result"})],
    ]
    cache = TTLCache(ttl_seconds=60)
    tool = SearchVideosTool(client=client, cache=cache)

    anon_result = tool.execute({"query": "python"})
    authed_result = tool.execute({"query": "python", "access_token": "tkn"})

    assert anon_result.data["results"][0]["video_id"] == "public-result"
    assert authed_result.data["results"][0]["video_id"] == "authed-result"
    assert client.search.call_count == 2

    # Repeating either call now hits its own cache entry, not the other one.
    anon_again = tool.execute({"query": "python"})
    assert anon_again.data["cached"] is True
    assert anon_again.data["results"][0]["video_id"] == "public-result"
    assert client.search.call_count == 2
