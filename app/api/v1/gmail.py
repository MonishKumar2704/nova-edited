"""
`/api/v1/gmail/...` (master spec section 45 API DESIGN, Phase 6/7).

Route handlers stay thin (master spec section 5): parse/validate the
request, resolve an OAuth access token, call the appropriate tool via the
registry, serialize the `ToolResult`. All Gmail API logic lives in
`app.integrations.gmail_api` / `app.tools.gmail.*`.

Unlike YouTube's search/get_video, every Gmail endpoint here requires a
connected Google account - there is no API-key/public-data fallback
(Gmail has no public data), so `_required_access_token()` is used
unconditionally rather than mirroring youtube.py's optional variant.

Phase 7 adds: message actions (mark read/unread, archive, trash/untrash,
star/unstar, label/unlabel), drafts (list/create/update/delete/send),
direct send, conversations (reply/reply-all/forward), and attachment
download. Every route that puts a message on the wire, or otherwise has a
Tool with `requires_confirmation = True`, returns the tool's `success`
flag as-is - the *enforcement* of "ask the user before executing" is the
agent orchestrator's job (Phase 9); these routes are the direct-call path
used once a caller (UI button, orchestrator) has already decided to
proceed.
"""

from __future__ import annotations

from flask import Blueprint, g, jsonify, request
from pydantic import ValidationError as PydanticValidationError

from app.api.v1._google_auth_helpers import registry as _registry
from app.api.v1._google_auth_helpers import require_access_token
from app.core.errors import ValidationError
from app.schemas.gmail import (
    GmailComposeRequest,
    GmailForwardRequest,
    GmailLabelActionRequest,
    GmailListMessagesRequest,
    GmailReplyRequest,
)
from app.services.dynamic_ui import (
    build_gmail_draft_cards,
    build_gmail_message_cards,
    build_gmail_thread_cards,
)

gmail_bp = Blueprint("gmail", __name__)


def _required_access_token() -> str:
    return require_access_token(
        "This action requires a connected Google account with Gmail access. "
        "Connect one via /api/v1/auth/google/connect."
    )


def _parse_label_ids() -> list[str] | None:
    raw = request.args.get("label_ids")
    if not raw:
        return None
    return [label.strip() for label in raw.split(",") if label.strip()]


@gmail_bp.route("/messages", methods=["GET"])
def list_messages():
    try:
        req = GmailListMessagesRequest(
            query=request.args.get("q"),
            label_ids=_parse_label_ids(),
            max_results=int(request.args.get("max_results", 25)),
            page_token=request.args.get("page_token"),
        )
    except (PydanticValidationError, ValueError) as exc:
        raise ValidationError("Invalid Gmail list-messages request.", details={"errors": str(exc)}) from exc

    tool = _registry().get("gmail.list_messages")
    result = tool.execute(
        {
            "access_token": _required_access_token(),
            "query": req.query,
            "label_ids": req.label_ids,
            "max_results": req.max_results,
            "page_token": req.page_token,
        }
    )
    messages = result.data.get("messages", [])
    return jsonify(
        {
            "success": result.success,
            "messages": messages,
            "cards": build_gmail_message_cards(messages),
            "next_page_token": result.data.get("next_page_token"),
            "request_id": getattr(g, "request_id", None),
        }
    )


@gmail_bp.route("/messages/<message_id>", methods=["GET"])
def get_message(message_id: str):
    tool = _registry().get("gmail.get_message")
    result = tool.execute(
        {
            "access_token": _required_access_token(),
            "message_id": message_id,
            "format": request.args.get("format", "full"),
        }
    )
    message = result.data.get("message")
    return jsonify(
        {
            "success": result.success,
            "message": message,
            "card": build_gmail_message_cards([message])[0] if message else None,
            "request_id": getattr(g, "request_id", None),
        }
    )


@gmail_bp.route("/threads", methods=["GET"])
def list_threads():
    tool = _registry().get("gmail.list_threads")
    result = tool.execute(
        {
            "access_token": _required_access_token(),
            "query": request.args.get("q"),
            "label_ids": _parse_label_ids(),
            "max_results": int(request.args.get("max_results", 25)),
            "page_token": request.args.get("page_token"),
        }
    )
    threads = result.data.get("threads", [])
    return jsonify(
        {
            "success": result.success,
            "threads": threads,
            "cards": build_gmail_thread_cards(threads),
            "next_page_token": result.data.get("next_page_token"),
            "request_id": getattr(g, "request_id", None),
        }
    )


@gmail_bp.route("/threads/<thread_id>", methods=["GET"])
def get_thread(thread_id: str):
    tool = _registry().get("gmail.get_thread")
    result = tool.execute({"access_token": _required_access_token(), "thread_id": thread_id})
    return jsonify(
        {
            "success": result.success,
            "thread": result.data.get("thread"),
            "request_id": getattr(g, "request_id", None),
        }
    )


@gmail_bp.route("/labels", methods=["GET"])
def list_labels():
    tool = _registry().get("gmail.list_labels")
    result = tool.execute({"access_token": _required_access_token()})
    return jsonify(
        {
            "success": result.success,
            "labels": result.data.get("labels", []),
            "request_id": getattr(g, "request_id", None),
        }
    )


# -- Phase 7: message actions ------------------------------------------


def _message_action_response(tool_name: str, message_id: str):
    tool = _registry().get(tool_name)
    result = tool.execute({"message_id": message_id, "access_token": _required_access_token()})
    message = result.data.get("message")
    return jsonify(
        {
            "success": result.success,
            "message": message,
            "card": build_gmail_message_cards([message])[0] if message else None,
            "request_id": getattr(g, "request_id", None),
        }
    )


@gmail_bp.route("/messages/<message_id>/mark_read", methods=["POST"])
def mark_read(message_id: str):
    return _message_action_response("gmail.mark_read", message_id)


@gmail_bp.route("/messages/<message_id>/mark_unread", methods=["POST"])
def mark_unread(message_id: str):
    return _message_action_response("gmail.mark_unread", message_id)


@gmail_bp.route("/messages/<message_id>/archive", methods=["POST"])
def archive_message(message_id: str):
    return _message_action_response("gmail.archive", message_id)


@gmail_bp.route("/messages/<message_id>/trash", methods=["POST"])
def trash_message(message_id: str):
    return _message_action_response("gmail.trash", message_id)


@gmail_bp.route("/messages/<message_id>/untrash", methods=["POST"])
def untrash_message(message_id: str):
    return _message_action_response("gmail.untrash", message_id)


@gmail_bp.route("/messages/<message_id>/star", methods=["POST"])
def star_message(message_id: str):
    return _message_action_response("gmail.star", message_id)


@gmail_bp.route("/messages/<message_id>/unstar", methods=["POST"])
def unstar_message(message_id: str):
    return _message_action_response("gmail.unstar", message_id)


@gmail_bp.route("/messages/<message_id>/labels", methods=["POST"])
def add_label(message_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        req = GmailLabelActionRequest(**payload)
    except PydanticValidationError as exc:
        raise ValidationError("Invalid add-label request.", details={"errors": str(exc)}) from exc

    tool = _registry().get("gmail.add_label")
    result = tool.execute({"message_id": message_id, "label_id": req.label_id, "access_token": _required_access_token()})
    message = result.data.get("message")
    return jsonify(
        {
            "success": result.success,
            "message": message,
            "card": build_gmail_message_cards([message])[0] if message else None,
            "request_id": getattr(g, "request_id", None),
        }
    )


@gmail_bp.route("/messages/<message_id>/labels/<label_id>", methods=["DELETE"])
def remove_label(message_id: str, label_id: str):
    tool = _registry().get("gmail.remove_label")
    result = tool.execute({"message_id": message_id, "label_id": label_id, "access_token": _required_access_token()})
    message = result.data.get("message")
    return jsonify(
        {
            "success": result.success,
            "message": message,
            "card": build_gmail_message_cards([message])[0] if message else None,
            "request_id": getattr(g, "request_id", None),
        }
    )


# -- Phase 7: drafts ------------------------------------------------------


@gmail_bp.route("/drafts", methods=["GET"])
def list_drafts():
    tool = _registry().get("gmail.draft.list")
    result = tool.execute(
        {
            "access_token": _required_access_token(),
            "max_results": int(request.args.get("max_results", 25)),
            "page_token": request.args.get("page_token"),
        }
    )
    drafts = result.data.get("drafts", [])
    return jsonify(
        {
            "success": result.success,
            "drafts": drafts,
            "cards": build_gmail_draft_cards(drafts),
            "next_page_token": result.data.get("next_page_token"),
            "request_id": getattr(g, "request_id", None),
        }
    )


def _parse_compose_request() -> GmailComposeRequest:
    payload = request.get_json(silent=True) or {}
    try:
        return GmailComposeRequest(**payload)
    except PydanticValidationError as exc:
        raise ValidationError("Invalid Gmail compose request.", details={"errors": str(exc)}) from exc


@gmail_bp.route("/drafts", methods=["POST"])
def create_draft():
    req = _parse_compose_request()
    tool = _registry().get("gmail.draft.create")
    result = tool.execute(
        {
            "to": req.to,
            "subject": req.subject,
            "body_text": req.body_text,
            "cc": req.cc,
            "bcc": req.bcc,
            "access_token": _required_access_token(),
        }
    )
    draft = result.data.get("draft")
    return (
        jsonify(
            {
                "success": result.success,
                "draft": draft,
                "card": build_gmail_draft_cards([draft])[0] if draft else None,
                "request_id": getattr(g, "request_id", None),
            }
        ),
        201,
    )


@gmail_bp.route("/drafts/<draft_id>", methods=["PUT"])
def update_draft(draft_id: str):
    req = _parse_compose_request()
    tool = _registry().get("gmail.draft.update")
    result = tool.execute(
        {
            "draft_id": draft_id,
            "to": req.to,
            "subject": req.subject,
            "body_text": req.body_text,
            "cc": req.cc,
            "bcc": req.bcc,
            "access_token": _required_access_token(),
        }
    )
    draft = result.data.get("draft")
    return jsonify(
        {
            "success": result.success,
            "draft": draft,
            "card": build_gmail_draft_cards([draft])[0] if draft else None,
            "request_id": getattr(g, "request_id", None),
        }
    )


@gmail_bp.route("/drafts/<draft_id>", methods=["DELETE"])
def delete_draft(draft_id: str):
    tool = _registry().get("gmail.draft.delete")
    result = tool.execute({"draft_id": draft_id, "access_token": _required_access_token()})
    return jsonify({"success": result.success, "deleted": result.data.get("deleted", False), "request_id": getattr(g, "request_id", None)})


@gmail_bp.route("/drafts/<draft_id>/send", methods=["POST"])
def send_draft(draft_id: str):
    tool = _registry().get("gmail.draft.send")
    result = tool.execute({"draft_id": draft_id, "access_token": _required_access_token()})
    message = result.data.get("message")
    return jsonify(
        {
            "success": result.success,
            "message": message,
            "card": build_gmail_message_cards([message])[0] if message else None,
            "request_id": getattr(g, "request_id", None),
        }
    )


# -- Phase 7: direct send ---------------------------------------------------


@gmail_bp.route("/messages/send", methods=["POST"])
def send_message():
    req = _parse_compose_request()
    tool = _registry().get("gmail.send")
    result = tool.execute(
        {
            "to": req.to,
            "subject": req.subject,
            "body_text": req.body_text,
            "cc": req.cc,
            "bcc": req.bcc,
            "access_token": _required_access_token(),
        }
    )
    message = result.data.get("message")
    return jsonify(
        {
            "success": result.success,
            "message": message,
            "card": build_gmail_message_cards([message])[0] if message else None,
            "request_id": getattr(g, "request_id", None),
        }
    )


# -- Phase 7: conversations -------------------------------------------------


@gmail_bp.route("/messages/<message_id>/reply", methods=["POST"])
def reply_to_message(message_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        req = GmailReplyRequest(**payload)
    except PydanticValidationError as exc:
        raise ValidationError("Invalid reply request.", details={"errors": str(exc)}) from exc

    tool = _registry().get("gmail.reply")
    result = tool.execute({"message_id": message_id, "body_text": req.body_text, "access_token": _required_access_token()})
    message = result.data.get("message")
    return jsonify(
        {
            "success": result.success,
            "message": message,
            "card": build_gmail_message_cards([message])[0] if message else None,
            "request_id": getattr(g, "request_id", None),
        }
    )


@gmail_bp.route("/messages/<message_id>/reply_all", methods=["POST"])
def reply_all_to_message(message_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        req = GmailReplyRequest(**payload)
    except PydanticValidationError as exc:
        raise ValidationError("Invalid reply-all request.", details={"errors": str(exc)}) from exc

    tool = _registry().get("gmail.reply_all")
    result = tool.execute(
        {
            "message_id": message_id,
            "body_text": req.body_text,
            "extra_cc": req.extra_cc,
            "access_token": _required_access_token(),
        }
    )
    message = result.data.get("message")
    return jsonify(
        {
            "success": result.success,
            "message": message,
            "card": build_gmail_message_cards([message])[0] if message else None,
            "request_id": getattr(g, "request_id", None),
        }
    )


@gmail_bp.route("/messages/<message_id>/forward", methods=["POST"])
def forward_message(message_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        req = GmailForwardRequest(**payload)
    except PydanticValidationError as exc:
        raise ValidationError("Invalid forward request.", details={"errors": str(exc)}) from exc

    tool = _registry().get("gmail.forward")
    result = tool.execute(
        {
            "message_id": message_id,
            "to": req.to,
            "body_text": req.body_text,
            "access_token": _required_access_token(),
        }
    )
    message = result.data.get("message")
    return jsonify(
        {
            "success": result.success,
            "message": message,
            "card": build_gmail_message_cards([message])[0] if message else None,
            "request_id": getattr(g, "request_id", None),
        }
    )


# -- Phase 7: attachments -----------------------------------------------


@gmail_bp.route("/messages/<message_id>/attachments/<attachment_id>", methods=["GET"])
def get_attachment(message_id: str, attachment_id: str):
    tool = _registry().get("gmail.get_attachment")
    result = tool.execute(
        {"message_id": message_id, "attachment_id": attachment_id, "access_token": _required_access_token()}
    )
    return jsonify(
        {
            "success": result.success,
            "attachment": result.data.get("attachment"),
            "request_id": getattr(g, "request_id", None),
        }
    )
