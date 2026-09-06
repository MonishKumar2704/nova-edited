"""End-to-end tests for `/api/v1/gmail/...` (Task 19: FULL GMAIL TEST).

Mirrors the pattern already used by `tests/integration/test_youtube_api.py`:
route-level tests through the real Flask `client` fixture, with
`GmailApiClient` methods monkeypatched so no real Gmail API call is made.
`test_gmail_*` unit tests already cover each `Tool` in isolation in depth;
this file's job is the thinner "does the wiring behind each *route* work"
layer: auth enforcement, request parsing, response shape, and status
codes - the part that only exists once a route module, schema, and tool
are actually glued together, which no single unit test file exercises.
"""

from app.integrations.gmail_api import AttachmentData, DraftSummary, MessageDetail

MESSAGE = MessageDetail(
    message_id="m1",
    thread_id="t1",
    label_ids=("INBOX", "UNREAD"),
    snippet="hello",
    subject="Hi",
    from_="alice@example.com",
    to="bob@example.com",
    date="2026-01-01T00:00:00Z",
    is_unread=True,
    has_attachments=False,
    body_text="hello world",
)

DRAFT = DraftSummary(draft_id="d1", message=MESSAGE)


def _fake_token(monkeypatch, token="fake-token"):
    monkeypatch.setattr("app.api.v1.gmail.require_access_token", lambda message: token)


# -- Auth enforcement (every Gmail route requires a connected account,
#    unlike YouTube's optional-API-key fallback) --------------------------


def test_list_messages_without_connected_account_returns_401(client):
    resp = client.get("/api/v1/gmail/messages")
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "AUTHENTICATION_ERROR"


def test_send_message_without_connected_account_returns_401(client):
    resp = client.post("/api/v1/gmail/messages/send", json={"to": ["a@example.com"], "body_text": "hi"})
    assert resp.status_code == 401


# -- Read operations -------------------------------------------------------


def test_list_messages_endpoint_returns_messages_and_cards(client, monkeypatch):
    _fake_token(monkeypatch)
    monkeypatch.setattr(
        "app.integrations.gmail_api.GmailApiClient.list_messages",
        lambda self, **kwargs: ([MESSAGE], None),
    )
    resp = client.get("/api/v1/gmail/messages")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["messages"][0]["message_id"] == "m1"
    assert body["cards"][0]["type"] == "gmail_message"
    assert body["request_id"] is not None


def test_get_message_endpoint_returns_message_and_card(client, monkeypatch):
    _fake_token(monkeypatch)
    monkeypatch.setattr("app.integrations.gmail_api.GmailApiClient.get_message", lambda self, **kwargs: MESSAGE)
    resp = client.get("/api/v1/gmail/messages/m1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["message"]["message_id"] == "m1"
    assert body["card"]["type"] == "gmail_message"


def test_list_labels_endpoint(client, monkeypatch):
    _fake_token(monkeypatch)
    monkeypatch.setattr("app.integrations.gmail_api.GmailApiClient.list_labels", lambda self, **kwargs: [])
    resp = client.get("/api/v1/gmail/labels")
    assert resp.status_code == 200
    assert resp.get_json()["labels"] == []


# -- Message actions ---------------------------------------------------


def test_mark_read_endpoint(client, monkeypatch):
    _fake_token(monkeypatch)
    monkeypatch.setattr("app.integrations.gmail_api.GmailApiClient.modify_message", lambda self, **kwargs: MESSAGE)
    resp = client.post("/api/v1/gmail/messages/m1/mark_read")
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True


def test_archive_endpoint_without_connected_account_returns_401(client):
    resp = client.post("/api/v1/gmail/messages/m1/archive")
    assert resp.status_code == 401


def test_add_label_endpoint_requires_label_id(client, monkeypatch):
    _fake_token(monkeypatch)
    resp = client.post("/api/v1/gmail/messages/m1/labels", json={})
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "VALIDATION_ERROR"


def test_add_label_endpoint_success(client, monkeypatch):
    _fake_token(monkeypatch)
    monkeypatch.setattr("app.integrations.gmail_api.GmailApiClient.modify_message", lambda self, **kwargs: MESSAGE)
    resp = client.post("/api/v1/gmail/messages/m1/labels", json={"label_id": "IMPORTANT"})
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True


# -- Drafts ---------------------------------------------------------------


def test_list_drafts_endpoint(client, monkeypatch):
    _fake_token(monkeypatch)
    monkeypatch.setattr("app.integrations.gmail_api.GmailApiClient.list_drafts", lambda self, **kwargs: ([DRAFT], None))
    resp = client.get("/api/v1/gmail/drafts")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["drafts"][0]["draft_id"] == "d1"
    assert body["cards"][0]["type"] == "gmail_draft"


def test_create_draft_endpoint_requires_recipient(client, monkeypatch):
    _fake_token(monkeypatch)
    resp = client.post("/api/v1/gmail/drafts", json={"to": [], "body_text": "hi"})
    assert resp.status_code == 400


def test_create_draft_endpoint_success_returns_201(client, monkeypatch):
    _fake_token(monkeypatch)
    monkeypatch.setattr("app.integrations.gmail_api.GmailApiClient.create_draft", lambda self, **kwargs: DRAFT)
    resp = client.post("/api/v1/gmail/drafts", json={"to": ["a@example.com"], "body_text": "hi"})
    assert resp.status_code == 201
    assert resp.get_json()["draft"]["draft_id"] == "d1"


def test_delete_draft_endpoint(client, monkeypatch):
    _fake_token(monkeypatch)
    monkeypatch.setattr("app.integrations.gmail_api.GmailApiClient.delete_draft", lambda self, **kwargs: None)
    resp = client.delete("/api/v1/gmail/drafts/d1")
    assert resp.status_code == 200
    assert resp.get_json()["deleted"] is True


def test_send_draft_endpoint(client, monkeypatch):
    _fake_token(monkeypatch)
    monkeypatch.setattr("app.integrations.gmail_api.GmailApiClient.send_draft", lambda self, **kwargs: MESSAGE)
    resp = client.post("/api/v1/gmail/drafts/d1/send")
    assert resp.status_code == 200
    assert resp.get_json()["message"]["message_id"] == "m1"


# -- Direct send + AI-generated compose path --------------------------


def test_send_message_endpoint_success(client, monkeypatch):
    _fake_token(monkeypatch)
    monkeypatch.setattr("app.integrations.gmail_api.GmailApiClient.send_message", lambda self, **kwargs: MESSAGE)
    resp = client.post("/api/v1/gmail/messages/send", json={"to": ["a@example.com"], "body_text": "hi"})
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True


def test_send_message_endpoint_requires_at_least_one_recipient(client, monkeypatch):
    _fake_token(monkeypatch)
    resp = client.post("/api/v1/gmail/messages/send", json={"to": [], "body_text": "hi"})
    assert resp.status_code == 400


# -- Conversations: reply / reply-all / forward -------------------------


def test_reply_endpoint_requires_body_text(client, monkeypatch):
    _fake_token(monkeypatch)
    resp = client.post("/api/v1/gmail/messages/m1/reply", json={})
    assert resp.status_code == 400


def test_reply_endpoint_success(client, monkeypatch):
    _fake_token(monkeypatch)
    monkeypatch.setattr("app.tools.gmail.conversations.ReplyTool.execute", lambda self, params: _tool_result(MESSAGE))
    resp = client.post("/api/v1/gmail/messages/m1/reply", json={"body_text": "thanks!"})
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True


def test_forward_endpoint_requires_recipient(client, monkeypatch):
    _fake_token(monkeypatch)
    resp = client.post("/api/v1/gmail/messages/m1/forward", json={"to": []})
    assert resp.status_code == 400


def test_forward_endpoint_success(client, monkeypatch):
    _fake_token(monkeypatch)
    monkeypatch.setattr("app.tools.gmail.conversations.ForwardTool.execute", lambda self, params: _tool_result(MESSAGE))
    resp = client.post("/api/v1/gmail/messages/m1/forward", json={"to": ["c@example.com"]})
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True


# -- Attachments ------------------------------------------------------


def test_get_attachment_endpoint(client, monkeypatch):
    _fake_token(monkeypatch)
    attachment = AttachmentData(attachment_id="att1", size=123, data_base64="aGVsbG8=")
    monkeypatch.setattr("app.integrations.gmail_api.GmailApiClient.get_attachment", lambda self, **kwargs: attachment)
    resp = client.get("/api/v1/gmail/messages/m1/attachments/att1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["attachment"]["attachment_id"] == "att1"
    assert body["attachment"]["data_base64"] == "aGVsbG8="


def test_get_attachment_endpoint_without_connected_account_returns_401(client):
    resp = client.get("/api/v1/gmail/messages/m1/attachments/att1")
    assert resp.status_code == 401


def _tool_result(message):
    from app.tools.base import ToolResult

    return ToolResult(success=True, data={"message": message.to_dict()})
