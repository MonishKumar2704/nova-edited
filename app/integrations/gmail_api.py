"""
Official Gmail API v1 client (master spec section 19: GMAIL PRINCIPLE -
use the Gmail API, not `mailto:`/compose-URL hacks, as the primary
architecture).

Same shape as `app.integrations.youtube_api`: a thin `requests`-based REST
wrapper with zero Flask/session/tool-registry knowledge, so orchestration
(argument validation, ToolResult shaping, dynamic-UI cards) stays in
`app.tools.gmail.*` / `app.services.dynamic_ui`.

Auth model
----------
Unlike YouTube's public search, every Gmail endpoint is inherently
account-specific - there is no API-key path. All methods here require an
OAuth access token with at least the `gmail.readonly` scope (see
`app.core.config.Config.google_scopes`); write/send scopes are added in
Phase 7 alongside the tools that need them.

Phase 6 scope: read-only foundation (list/get messages, threads, labels).
Phase 7 adds message actions (mark read/unread, archive, trash, star,
label/unlabel - all via the single `messages.modify` endpoint, see
`modify_message`), composition (drafts, compose, send), conversations
(reply, reply-all, forward - built on `send_message`/`create_draft` with
proper `threadId`/`In-Reply-To`/`References` headers), and attachment
download.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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

API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
_REQUEST_TIMEOUT_SECONDS = 10
_METADATA_HEADERS = ["Subject", "From", "To", "Cc", "Date", "Message-ID"]


@dataclass(frozen=True)
class AttachmentMetadata:
    attachment_id: str
    filename: str
    mime_type: str
    size: int | None

    def to_dict(self) -> dict:
        return {
            "attachment_id": self.attachment_id,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size": self.size,
        }


@dataclass(frozen=True)
class MessageSummary:
    """Metadata-level view of a message (headers + snippet, no body)."""

    message_id: str
    thread_id: str
    label_ids: tuple[str, ...]
    snippet: str
    subject: str
    from_: str
    to: str
    date: str
    is_unread: bool
    has_attachments: bool

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "label_ids": list(self.label_ids),
            "snippet": self.snippet,
            "subject": self.subject,
            "from": self.from_,
            "to": self.to,
            "date": self.date,
            "is_unread": self.is_unread,
            "has_attachments": self.has_attachments,
        }


@dataclass(frozen=True)
class MessageDetail(MessageSummary):
    """Full view of a message: headers + body + attachment metadata."""

    body_text: str = ""
    body_html: str = ""
    attachments: tuple[AttachmentMetadata, ...] = field(default_factory=tuple)
    # The RFC 5322 `Message-ID` header (distinct from Gmail's own opaque
    # `message_id`) - needed to set `In-Reply-To`/`References` correctly
    # when replying/forwarding (see `app.tools.gmail.conversations`).
    rfc_message_id: str = ""

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "body_text": self.body_text,
            "body_html": self.body_html,
            "attachments": [a.to_dict() for a in self.attachments],
            "rfc_message_id": self.rfc_message_id,
        }


@dataclass(frozen=True)
class ThreadSummary:
    thread_id: str
    snippet: str
    message_count: int

    def to_dict(self) -> dict:
        return {"thread_id": self.thread_id, "snippet": self.snippet, "message_count": self.message_count}


@dataclass(frozen=True)
class ThreadDetail:
    thread_id: str
    messages: tuple[MessageDetail, ...]

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "messages": [m.to_dict() for m in self.messages],
            "message_count": len(self.messages),
        }


@dataclass(frozen=True)
class LabelSummary:
    label_id: str
    name: str
    type: str  # "system" | "user"
    messages_total: int | None
    messages_unread: int | None

    def to_dict(self) -> dict:
        return {
            "label_id": self.label_id,
            "name": self.name,
            "type": self.type,
            "messages_total": self.messages_total,
            "messages_unread": self.messages_unread,
        }


@dataclass(frozen=True)
class DraftSummary:
    """A saved draft: its own `draft_id` plus the `MessageDetail` it wraps."""

    draft_id: str
    message: MessageDetail

    def to_dict(self) -> dict:
        return {"draft_id": self.draft_id, "message": self.message.to_dict()}


@dataclass(frozen=True)
class AttachmentData:
    attachment_id: str
    size: int | None
    data_base64: str  # raw base64 (already re-encoded from Gmail's base64url), safe to hand to a client

    def to_dict(self) -> dict:
        return {"attachment_id": self.attachment_id, "size": self.size, "data_base64": self.data_base64}


class GmailApiClient:
    def __init__(self) -> None:
        # No API-key configuration knob (unlike YouTubeApiClient): Gmail
        # has no public-data path, so there is nothing to be "configured"
        # with beyond a per-request OAuth access token.
        pass

    # -- Messages (Phase 6: list/search/get) -----------------------------

    def list_messages(
        self,
        *,
        access_token: str,
        query: str | None = None,
        label_ids: list[str] | None = None,
        max_results: int = 25,
        page_token: str | None = None,
    ) -> tuple[list[MessageSummary], str | None]:
        """List message IDs matching `query`/`label_ids`, then resolve each to metadata.

        `messages.list` itself only returns `{id, threadId}` pairs - Gmail
        has no "list with headers/snippet inlined" endpoint - so each ID on
        the page is resolved via `get_message(format="metadata")`. Kept
        sequential (no batch-API dependency) since this is the Phase 6
        foundation; `max_results` is capped modestly for that reason.
        """
        params: dict = {"maxResults": max(1, min(max_results, 50))}
        if query:
            params["q"] = query
        if label_ids:
            params["labelIds"] = label_ids
        if page_token:
            params["pageToken"] = page_token

        body = self._get("/messages", params=params, access_token=access_token)
        refs = body.get("messages", [])
        summaries = [
            self.get_message(access_token=access_token, message_id=ref["id"], format="metadata")
            for ref in refs
            if ref.get("id")
        ]
        return summaries, body.get("nextPageToken")

    def get_message(self, *, access_token: str, message_id: str, format: str = "full") -> MessageDetail:
        params: dict = {"format": format}
        if format == "metadata":
            params["metadataHeaders"] = _METADATA_HEADERS
        body = self._get(f"/messages/{message_id}", params=params, access_token=access_token)
        return self._message_from_item(body)

    # -- Threads (Phase 6) ------------------------------------------------

    def list_threads(
        self,
        *,
        access_token: str,
        query: str | None = None,
        label_ids: list[str] | None = None,
        max_results: int = 25,
        page_token: str | None = None,
    ) -> tuple[list[ThreadSummary], str | None]:
        params: dict = {"maxResults": max(1, min(max_results, 50))}
        if query:
            params["q"] = query
        if label_ids:
            params["labelIds"] = label_ids
        if page_token:
            params["pageToken"] = page_token

        body = self._get("/threads", params=params, access_token=access_token)
        threads = [
            ThreadSummary(
                thread_id=item.get("id", ""),
                snippet=item.get("snippet", ""),
                message_count=len(item.get("messages") or []) or 1,
            )
            for item in body.get("threads", [])
        ]
        return threads, body.get("nextPageToken")

    def get_thread(self, *, access_token: str, thread_id: str, format: str = "full") -> ThreadDetail:
        params: dict = {"format": format}
        if format == "metadata":
            params["metadataHeaders"] = _METADATA_HEADERS
        body = self._get(f"/threads/{thread_id}", params=params, access_token=access_token)
        messages = tuple(self._message_from_item(item) for item in body.get("messages", []))
        return ThreadDetail(thread_id=body.get("id", thread_id), messages=messages)

    # -- Labels (Phase 6) ---------------------------------------------------

    def list_labels(self, *, access_token: str) -> list[LabelSummary]:
        body = self._get("/labels", params={}, access_token=access_token)
        labels = []
        for item in body.get("labels", []):
            labels.append(
                LabelSummary(
                    label_id=item.get("id", ""),
                    name=item.get("name", ""),
                    type=item.get("type", "user"),
                    messages_total=item.get("messagesTotal"),
                    messages_unread=item.get("messagesUnread"),
                )
            )
        return labels

    # -- Message actions (Phase 7: mark read/unread, archive, trash, star, labels) --
    #
    # Gmail exposes all of these through one endpoint, `messages.modify`,
    # by adding/removing label IDs - there is no separate "archive" or
    # "star" verb server-side. `_MODIFY_LABEL_ACTIONS` below maps each
    # user-facing action to the label delta so `app.tools.gmail.actions`
    # stays a thin translation layer with no Gmail-label knowledge of its
    # own beyond that mapping.

    def modify_message(
        self, *, access_token: str, message_id: str, add_label_ids: list[str] | None = None, remove_label_ids: list[str] | None = None
    ) -> MessageDetail:
        body: dict = {}
        if add_label_ids:
            body["addLabelIds"] = add_label_ids
        if remove_label_ids:
            body["removeLabelIds"] = remove_label_ids
        item = self._post(f"/messages/{message_id}/modify", json_body=body, access_token=access_token)
        return self._message_from_item(item)

    def trash_message(self, *, access_token: str, message_id: str) -> MessageDetail:
        item = self._post(f"/messages/{message_id}/trash", json_body=None, access_token=access_token)
        return self._message_from_item(item)

    def untrash_message(self, *, access_token: str, message_id: str) -> MessageDetail:
        item = self._post(f"/messages/{message_id}/untrash", json_body=None, access_token=access_token)
        return self._message_from_item(item)

    # -- Composition (Phase 7: drafts, compose, send) ------------------------

    def list_drafts(self, *, access_token: str, max_results: int = 25, page_token: str | None = None) -> tuple[list[DraftSummary], str | None]:
        params: dict = {"maxResults": max(1, min(max_results, 50))}
        if page_token:
            params["pageToken"] = page_token
        body = self._get("/drafts", params=params, access_token=access_token)
        drafts = [
            self.get_draft(access_token=access_token, draft_id=ref["id"])
            for ref in body.get("drafts", [])
            if ref.get("id")
        ]
        return drafts, body.get("nextPageToken")

    def get_draft(self, *, access_token: str, draft_id: str) -> DraftSummary:
        item = self._get(f"/drafts/{draft_id}", params={"format": "full"}, access_token=access_token)
        message = self._message_from_item(item.get("message", {}))
        return DraftSummary(draft_id=item.get("id", draft_id), message=message)

    def create_draft(
        self, *, access_token: str, to: list[str], subject: str, body_text: str, cc: list[str] | None = None,
        bcc: list[str] | None = None, thread_id: str | None = None, in_reply_to: str | None = None,
        references: str | None = None,
    ) -> DraftSummary:
        raw = _build_raw_mime(
            to=to, subject=subject, body_text=body_text, cc=cc, bcc=bcc, in_reply_to=in_reply_to, references=references
        )
        message_body: dict = {"raw": raw}
        if thread_id:
            message_body["threadId"] = thread_id
        item = self._post("/drafts", json_body={"message": message_body}, access_token=access_token)
        message = self._message_from_item(item.get("message", {}))
        return DraftSummary(draft_id=item.get("id", ""), message=message)

    def update_draft(
        self, *, access_token: str, draft_id: str, to: list[str], subject: str, body_text: str,
        cc: list[str] | None = None, bcc: list[str] | None = None, thread_id: str | None = None,
    ) -> DraftSummary:
        raw = _build_raw_mime(to=to, subject=subject, body_text=body_text, cc=cc, bcc=bcc)
        message_body: dict = {"raw": raw}
        if thread_id:
            message_body["threadId"] = thread_id
        item = self._put(f"/drafts/{draft_id}", json_body={"message": message_body}, access_token=access_token)
        message = self._message_from_item(item.get("message", {}))
        return DraftSummary(draft_id=item.get("id", draft_id), message=message)

    def delete_draft(self, *, access_token: str, draft_id: str) -> None:
        self._delete(f"/drafts/{draft_id}", params={}, access_token=access_token)

    def send_draft(self, *, access_token: str, draft_id: str) -> MessageDetail:
        item = self._post("/drafts/send", json_body={"id": draft_id}, access_token=access_token)
        return self._message_from_item(item)

    def send_message(
        self, *, access_token: str, to: list[str], subject: str, body_text: str, cc: list[str] | None = None,
        bcc: list[str] | None = None, thread_id: str | None = None, in_reply_to: str | None = None,
        references: str | None = None,
    ) -> MessageDetail:
        """Compose-and-send in one call (used for direct sends, reply/reply-all/forward)."""
        raw = _build_raw_mime(
            to=to, subject=subject, body_text=body_text, cc=cc, bcc=bcc, in_reply_to=in_reply_to, references=references
        )
        message_body: dict = {"raw": raw}
        if thread_id:
            message_body["threadId"] = thread_id
        item = self._post("/messages/send", json_body=message_body, access_token=access_token)
        return self._message_from_item(item)

    # -- Attachments (Phase 7: download; metadata already surfaced in Phase 6) --

    def get_attachment(self, *, access_token: str, message_id: str, attachment_id: str) -> AttachmentData:
        body = self._get(f"/messages/{message_id}/attachments/{attachment_id}", params={}, access_token=access_token)
        data_urlsafe = body.get("data", "")
        # Gmail returns attachment bytes base64url-encoded; re-encode to
        # standard base64 so this is directly usable by any client (browser
        # `atob`, email libraries, etc) without a urlsafe-decode step.
        padded = data_urlsafe + "=" * (-len(data_urlsafe) % 4)
        raw_bytes = base64.urlsafe_b64decode(padded) if data_urlsafe else b""
        return AttachmentData(
            attachment_id=body.get("attachmentId", attachment_id),
            size=body.get("size"),
            data_base64=base64.b64encode(raw_bytes).decode("ascii"),
        )

    # -- internal (writes) ---------------------------------------------------

    def _post(self, path: str, *, json_body: dict | None, access_token: str) -> dict:
        return self._write("post", path, json_body=json_body, access_token=access_token)

    def _put(self, path: str, *, json_body: dict | None, access_token: str) -> dict:
        return self._write("put", path, json_body=json_body, access_token=access_token)

    def _delete(self, path: str, *, params: dict, access_token: str) -> None:
        self._write("delete", path, json_body=None, access_token=access_token, expect_json=False)

    def _write(self, method: str, path: str, *, json_body: dict | None, access_token: str, expect_json: bool = True) -> dict:
        if not access_token:
            raise AuthenticationError(
                "This action requires a connected Google account with Gmail access. "
                "Connect one via /api/v1/auth/google/connect."
            )
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            resp = requests.request(
                method, f"{API_BASE}{path}", json=json_body, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS
            )
        except requests.Timeout as exc:
            raise TimeoutErrorNova("Timed out talking to the Gmail API.") from exc
        except requests.RequestException as exc:
            raise NetworkError("Could not reach the Gmail API.") from exc

        if resp.status_code == 401:
            raise AuthenticationError("Gmail rejected the request credentials (expired/invalid token).")
        if resp.status_code == 403:
            raise _classify_403(resp)
        if resp.status_code == 404:
            raise NotFoundError(f"Gmail resource not found: {_extract_error_message(resp)}")
        if resp.status_code == 400:
            raise ValidationError(f"Gmail API rejected the request: {_extract_error_message(resp)}")
        if not resp.ok:
            raise NetworkError(f"Gmail API returned HTTP {resp.status_code}.")

        if not expect_json or not resp.content:
            return {}
        return resp.json()

    # -- internal (reads) ----------------------------------------------------

    def _message_from_item(self, item: dict) -> MessageDetail:
        payload = item.get("payload", {}) or {}
        # Header field names are case-insensitive per RFC 5322 section
        # 2.2 - some senders don't send them Title-Cased ("subject"/"from"
        # rather than "Subject"/"From"), and Gmail passes the raw casing
        # through unchanged. Normalize to lower-case on both sides so a
        # message from such a sender still gets a subject/from/to/date
        # instead of silently reading as blank.
        headers = {h.get("name", "").lower(): h.get("value", "") for h in payload.get("headers", [])}
        label_ids = tuple(item.get("labelIds", []) or [])
        body_text, body_html, attachments = _walk_parts(payload)
        return MessageDetail(
            message_id=item.get("id", ""),
            thread_id=item.get("threadId", ""),
            label_ids=label_ids,
            snippet=item.get("snippet", ""),
            subject=headers.get("subject", ""),
            from_=headers.get("from", ""),
            to=headers.get("to", ""),
            date=headers.get("date", ""),
            is_unread="UNREAD" in label_ids,
            has_attachments=bool(attachments),
            body_text=body_text,
            body_html=body_html,
            attachments=tuple(attachments),
            rfc_message_id=headers.get("message-id", ""),
        )

    def _get(self, path: str, *, params: dict, access_token: str) -> dict:
        if not access_token:
            raise AuthenticationError(
                "This action requires a connected Google account with Gmail access. "
                "Connect one via /api/v1/auth/google/connect."
            )
        headers = {"Authorization": f"Bearer {access_token}"}

        try:
            resp = requests.get(f"{API_BASE}{path}", params=params, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS)
        except requests.Timeout as exc:
            raise TimeoutErrorNova("Timed out talking to the Gmail API.") from exc
        except requests.RequestException as exc:
            raise NetworkError("Could not reach the Gmail API.") from exc

        if resp.status_code == 401:
            raise AuthenticationError("Gmail rejected the request credentials (expired/invalid token).")
        if resp.status_code == 403:
            raise _classify_403(resp)
        if resp.status_code == 404:
            raise NotFoundError(f"Gmail resource not found: {_extract_error_message(resp)}")
        if resp.status_code == 400:
            raise ValidationError(f"Gmail API rejected the request: {_extract_error_message(resp)}")
        if not resp.ok:
            raise NetworkError(f"Gmail API returned HTTP {resp.status_code}.")

        return resp.json()


def _walk_parts(payload: dict) -> tuple[str, str, list[AttachmentMetadata]]:
    """Recursively walk a message `payload`, extracting text/html bodies + attachment metadata.

    Gmail's MIME tree can nest arbitrarily (multipart/mixed containing
    multipart/alternative containing text/plain + text/html, etc), so this
    recurses through `parts` rather than assuming a fixed depth.
    """
    body_text = ""
    body_html = ""
    attachments: list[AttachmentMetadata] = []

    mime_type = payload.get("mimeType", "")
    filename = payload.get("filename", "") or ""
    body = payload.get("body", {}) or {}

    if filename and body.get("attachmentId"):
        attachments.append(
            AttachmentMetadata(
                attachment_id=body["attachmentId"],
                filename=filename,
                mime_type=mime_type,
                size=body.get("size"),
            )
        )
    elif mime_type == "text/plain" and body.get("data"):
        body_text += _decode_base64url(body["data"])
    elif mime_type == "text/html" and body.get("data"):
        body_html += _decode_base64url(body["data"])

    for part in payload.get("parts", []) or []:
        part_text, part_html, part_attachments = _walk_parts(part)
        body_text += part_text
        body_html += part_html
        attachments.extend(part_attachments)

    return body_text, body_html, attachments


def _build_raw_mime(
    *,
    to: list[str],
    subject: str,
    body_text: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    in_reply_to: str | None = None,
    references: str | None = None,
) -> str:
    """Build a base64url `raw` RFC 2822 message for `messages.send`/`drafts.create`.

    Plain `text/plain` only (master spec keeps this to what's actually
    needed: AI-authored email bodies, not arbitrary HTML). `In-Reply-To`/
    `References` are set for reply/forward so Gmail (and other clients)
    thread the message correctly even when `threadId` alone would be
    ambiguous.
    """
    msg = MIMEMultipart()
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    if bcc:
        msg["Bcc"] = ", ".join(bcc)
    msg["Subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references
    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    return base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")


def _decode_base64url(data: str) -> str:
    try:
        padded = data + "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return ""


def _classify_403(resp: requests.Response) -> NovaError:
    message = _extract_error_message(resp)
    lowered = message.lower()
    if "quota" in lowered or "rate" in lowered:
        return RateLimitError(f"Gmail API quota exceeded: {message}")
    return AuthenticationError(f"Gmail API access forbidden: {message}")


def _extract_error_message(resp: requests.Response) -> str:
    try:
        body = resp.json()
        return body.get("error", {}).get("message", resp.text[:200])
    except ValueError:
        return resp.text[:200]
