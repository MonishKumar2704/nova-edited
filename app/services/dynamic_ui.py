"""
Dynamic UI card/action building (master spec section 10, Phase 4).

Backend tools/integrations return plain data (see `app.tools.youtube`,
`app.integrations.youtube_api`) with zero UI knowledge. This module is the
one place that wraps that data into the generic

    {"type": ..., "data": {...}, "actions": [{"id", "label",
     "requires_confirmation"}, ...]}

envelope the frontend renders generically (`static/js/dynamic-actions.js`).
Adding a new card type (e.g. a Gmail message card in Phase 6/7) means
adding one function here - the frontend's rendering code and the
orchestrator/API routes that call this module do not need to change.
"""

from __future__ import annotations

from typing import Any


def action(action_id: str, label: str, *, requires_confirmation: bool = False) -> dict[str, Any]:
    """A single dynamic action definition (master spec section 10)."""
    return {"id": action_id, "label": label, "requires_confirmation": requires_confirmation}


# Actions available on every YouTube video result card. Kept as a plain
# module-level constant (not per-video state) since none of these actions
# currently vary by video - `list(...)` is returned to callers so nobody
# accidentally mutates the shared list.
VIDEO_ACTIONS: list[dict[str, Any]] = [
    action("play", "Play"),
    action("open", "Open on YouTube"),
    action("queue", "Add to Queue"),
    action("add_to_playlist", "Add to Playlist", requires_confirmation=True),
    action("share", "Share"),
]


def build_video_card(video: dict[str, Any]) -> dict[str, Any]:
    """Wrap one video dict (from `VideoSummary`/`VideoDetails.to_dict()`) into a card."""
    return {"type": "youtube_video", "data": video, "actions": list(VIDEO_ACTIONS)}


def build_video_cards(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_video_card(video) for video in videos]


# --- Phase 5: playlists ------------------------------------------------

PLAYLIST_ACTIONS: list[dict[str, Any]] = [
    action("open", "View Playlist"),
    action("delete_playlist", "Delete Playlist", requires_confirmation=True),
]

# Actions on one video *within* a playlist (distinct from `VIDEO_ACTIONS`:
# "remove"/"reorder" only make sense in this context and need the
# playlist-item id, not just the video id).
PLAYLIST_ITEM_ACTIONS: list[dict[str, Any]] = [
    action("play", "Play"),
    action("open", "Open on YouTube"),
    action("remove_from_playlist", "Remove", requires_confirmation=True),
]


def build_playlist_card(playlist: dict[str, Any]) -> dict[str, Any]:
    """Wrap one playlist dict (from `PlaylistSummary.to_dict()`) into a card."""
    return {"type": "youtube_playlist", "data": playlist, "actions": list(PLAYLIST_ACTIONS)}


def build_playlist_cards(playlists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_playlist_card(p) for p in playlists]


def build_playlist_item_card(item: dict[str, Any]) -> dict[str, Any]:
    """Wrap one playlist-item dict (from `PlaylistItem.to_dict()`) into a card."""
    return {"type": "youtube_playlist_item", "data": item, "actions": list(PLAYLIST_ITEM_ACTIONS)}


def build_playlist_item_cards(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_playlist_item_card(item) for item in items]


# --- Phase 6/7: Gmail --------------------------------------------------
#
# Phase 6 shipped the read-only foundation (just "open"). Phase 7 adds
# message actions (mark read/unread, archive, trash, star/unstar) and
# conversation actions (reply, reply-all, forward) backed by
# `app.tools.gmail.actions`/`app.tools.gmail.conversations`. Sends have
# real, visible side effects (an email actually leaves the account), so
# reply/reply_all/forward are `requires_confirmation=True` - archive/trash
# are reversible (untrash/re-add INBOX) but still touch real account state,
# so they are too, matching the YouTube playlist-mutation precedent above.
# Mark read/unread and star/unstar are low-stakes and instantly reversible,
# so they are not.
#
# Both directions of the read and star toggles are listed (mark_read AND
# mark_unread; star AND unstar) rather than trying to show only the
# currently-applicable one, since this card is built the same way
# regardless of a message's current label state (Task 35: verify Gmail
# organization - `unstar` was previously missing here, so a card had no
# way to reach `gmail.unstar`, even though the tool/route existed).

GMAIL_MESSAGE_ACTIONS: list[dict[str, Any]] = [
    # Task 13: label reflects what this now actually does - open/read the
    # message inside Nova's own Gmail workspace, not launch
    # mail.google.com in another tab (see the `open` case in
    # app/templates/index.html's `handleCardAction`, which fetches the
    # full message via `/api/v1/gmail/messages/<id>` and renders it into
    # the internal reading pane).
    action("open", "Open"),
    action("reply", "Reply", requires_confirmation=True),
    action("reply_all", "Reply All", requires_confirmation=True),
    action("forward", "Forward", requires_confirmation=True),
    action("mark_read", "Mark Read"),
    action("mark_unread", "Mark Unread"),
    action("star", "Star"),
    action("unstar", "Unstar"),
    action("archive", "Archive", requires_confirmation=True),
    action("trash", "Delete", requires_confirmation=True),
]

GMAIL_THREAD_ACTIONS: list[dict[str, Any]] = [
    # Task 13: same fix as GMAIL_MESSAGE_ACTIONS above - opens the thread's
    # messages in Nova's internal reading pane.
    action("open", "Open"),
    action("reply", "Reply", requires_confirmation=True),
]

GMAIL_DRAFT_ACTIONS: list[dict[str, Any]] = [
    action("edit", "Edit Draft"),
    action("send", "Send", requires_confirmation=True),
    action("delete_draft", "Delete Draft", requires_confirmation=True),
]


def build_gmail_message_card(message: dict[str, Any]) -> dict[str, Any]:
    """Wrap one message dict (from `MessageSummary`/`MessageDetail.to_dict()`) into a card.

    Task 16: this used to also stamp a `mail.google.com` deep link onto
    `data["url"]`. Nothing reads it anymore - the card's "open" action
    (see GMAIL_MESSAGE_ACTIONS) has opened the message inside Nova's own
    reading pane since Task 13, and no test or frontend code looks at
    this field (checked: neither `dynamic-actions.js` nor
    `app/templates/index.html` reference a Gmail card's `data.url`).
    Leaving it in would just be an obsolete external-navigation target
    sitting unused in the payload, so it's dropped rather than kept
    "just in case".
    """
    return {"type": "gmail_message", "data": dict(message), "actions": list(GMAIL_MESSAGE_ACTIONS)}


def build_gmail_message_cards(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_gmail_message_card(m) for m in messages]


def build_gmail_thread_card(thread: dict[str, Any]) -> dict[str, Any]:
    """Wrap one thread dict (from `ThreadSummary`/`ThreadDetail.to_dict()`) into a card.

    Task 16: same cleanup as `build_gmail_message_card` above - the
    `mail.google.com` thread-link `url` field is dropped as dead,
    obsolete external-navigation data now that "open" reads the thread
    into Nova's own reading pane (Task 13).
    """
    return {"type": "gmail_thread", "data": dict(thread), "actions": list(GMAIL_THREAD_ACTIONS)}


def build_gmail_thread_cards(threads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_gmail_thread_card(t) for t in threads]


def build_gmail_draft_card(draft: dict[str, Any]) -> dict[str, Any]:
    """Wrap one draft dict (from `DraftSummary.to_dict()`) into a card."""
    return {"type": "gmail_draft", "data": draft, "actions": list(GMAIL_DRAFT_ACTIONS)}


def build_gmail_draft_cards(drafts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_gmail_draft_card(d) for d in drafts]
