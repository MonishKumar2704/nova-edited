def test_legacy_agent_youtube_command_without_api_key_still_returns_200(client):
    # No YOUTUBE_API_KEY is configured in the test environment and no
    # Google account is connected, so youtube.search fails closed with a
    # classified error - this must surface as a structured, non-500
    # response (not a crash), same contract as the old scraping fallback.
    resp = client.post("/agent", json={"command": "open youtube and play lofi beats"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is False
    assert "request_id" in body


def test_legacy_agent_youtube_command_plays_first_result(client, monkeypatch):
    from app.integrations.youtube_api import VideoSummary

    def fake_search(self, *, query, max_results, order, access_token):
        return [
            VideoSummary(
                video_id="abc123XYZ",
                title="Lofi Beats to Study To",
                channel_title="Chill Channel",
                description="...",
                thumbnail_url=None,
                published_at=None,
                url="https://www.youtube.com/watch?v=abc123XYZ",
            )
        ]

    monkeypatch.setattr("app.integrations.youtube_api.YouTubeApiClient.search", fake_search)

    resp = client.post("/agent", json={"command": "open youtube and play lofi beats"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert "abc123XYZ" in body["url"]

    # The versioned endpoint returns the full structured payload.
    resp_v1 = client.post("/api/v1/agent/command", json={"command": "open youtube and play lofi beats"})
    body_v1 = resp_v1.get_json()
    assert body_v1["action_type"] == "youtube_play"
    assert body_v1["data"]["video_id"] == "abc123XYZ"


def test_legacy_agent_gmail_command(client):
    # Task 16: this used to assert on a `mail.google.com` compose-window
    # deep link in `body["url"]` - that was the last obsolete Gmail
    # external-opening code path in the legacy fallback router, now
    # removed (see `_compose_or_draft`), so the response no longer
    # carries a `url` at all. What's left to check is that Nova still
    # understood the command (the recipient shows up in the message).
    resp = client.post("/agent", json={"command": "email john at gmail dot com type hello there"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["action_type"] == "gmail_compose"
    assert "john@gmail.com" in body["message"]
    assert not body.get("url")


def test_legacy_agent_missing_command_returns_validation_error(client):
    resp = client.post("/agent", json={})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_legacy_agent_unrecognized_command_no_longer_crashes(client):
    # Regression test for the original NameError-on-unmatched-command bug.
    resp = client.post("/agent", json={"command": "what is the weather today"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is False


def test_versioned_agent_command_endpoint(client):
    resp = client.post("/api/v1/agent/command", json={"command": "email jane at gmail dot com type hi"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["action_type"] == "gmail_compose"
    assert body["request_id"] is not None


def test_versioned_agent_command_validation_error(client):
    resp = client.post("/api/v1/agent/command", json={"command": ""})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_home_page_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Nova Voice Agent" in resp.data
