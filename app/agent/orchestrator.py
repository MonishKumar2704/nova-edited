"""
Agent orchestrator interface + the temporary legacy implementation.

`AgentOrchestrator` is the seam the API/service layer depends on.
`LegacyRuleBasedOrchestrator` reproduces the original project's
keyword-matching *routing* behavior (unchanged, so existing voice
commands keep working) but now lives in one place, is testable in
isolation, and returns a structured `AgentActionResult` instead of
ad-hoc dict/global juggling.

As of Phase 3, YouTube lookups go through the real `youtube.search` tool
(official YouTube Data API v3, via the `ToolRegistry`) instead of the
removed HTML-scraping fallback - the *routing* ("does this look like a
YouTube command?") is still keyword-based pending Phase 9's real
planner, but the *execution* is no longer a scraping hack.

As of Phase 9, this legacy router is no longer the primary orchestrator -
`app.agent.planner_orchestrator.PlannerOrchestrator` is - but it is kept
and wired in as that planner's graceful-degradation fallback (master spec
section 15: a session with no LLM provider configured/reachable must
still get basic YouTube/Gmail command handling, not a hard failure).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.core.errors import NovaError
from app.services.dynamic_ui import (
    build_gmail_draft_card,
    build_gmail_draft_cards,
    build_gmail_message_cards,
    build_video_card,
)
from app.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from app.agent.context import AgentRequestContext


@dataclass
class AgentActionResult:
    success: bool
    message: str
    url: str | None = None
    action_type: str = "unknown"
    # Structured payload + dynamic actions (master spec section 8/10). As
    # of Phase 4, YouTube results populate `data["card"]`/`actions` and the
    # frontend (`app/templates/index.html` + `static/js/dynamic-actions.js`)
    # renders them instead of only opening `url`. Gmail results (Phase 6/7)
    # will populate the same fields with a `gmail_message` card type.
    data: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)


class AgentOrchestrator(ABC):
    """Seam between the API layer and command interpretation/execution."""

    @abstractmethod
    def handle(self, command: str, *, context: "AgentRequestContext | None" = None) -> AgentActionResult:
        """Interpret and execute one user command.

        `context` (added in Phase 9; optional/ignored by implementations
        that don't need it, e.g. this module's legacy router) carries
        request-scoped data - session ID, an OAuth access token if the
        session has one, and whether this call is answering a pending
        confirmation. See `app.agent.context.AgentRequestContext`.
        """
        raise NotImplementedError


_YOUTUBE_STRIP_PATTERNS = [
    "open youtube and search",
    "open youtube and play",
    "open youtube",
    "and play",
    "play",
    "on youtube",
]

_YOUTUBE_TRIGGER_WORDS = ("youtube", "video", "channel", "playlist")
_GMAIL_TRIGGER_WORDS = ("gmail", "email", "mail", "message")

# Task 15 fix: bare Gmail action commands ("Archive this.", "Delete this.",
# "Star this.", "Mark this as read.", "Forward this email.") reference an
# already-open email/thread the way "reply to this email" already does, but
# unlike "reply to this email" they don't contain any word in
# `_GMAIL_TRIGGER_WORDS` ("gmail"/"email"/"mail"/"message" - "this."/"this
# email" alone can be missing "email" entirely, e.g. plain "Archive this."),
# so `handle()` was routing them straight to the generic "Command not
# understood" fallback instead of `_handle_gmail`'s existing "needs the AI
# assistant" messaging. This set lets `handle()` recognize them without
# touching `_GMAIL_TRIGGER_WORDS` itself (which stays as the broader/looser
# match used everywhere else).
_GMAIL_BARE_ACTION_WORDS = (
    "archive",
    "trash",
    "delete",
    "untrash",
    "restore",
    "star",
    "unstar",
    "forward",
    "reply",
)

# Task 15 fix: "Show my drafts.", "Edit this draft.", "Send this draft."
# reference Gmail drafts but, like the bare actions above, contain none of
# `_GMAIL_TRIGGER_WORDS` - "draft" isn't "gmail"/"email"/"mail"/"message" -
# so they also need their own top-level trigger check.
_GMAIL_DRAFT_TRIGGER_WORDS = ("draft",)

# A stripped-down query that's just a bare reference ("this video", "that
# channel", "it") can't be resolved without conversation memory - this
# router has none (see class docstring), so these are recognized and
# answered with a plain "needs the AI assistant" message instead of being
# sent to `youtube.search` as a literal, nonsensical query string.
_BARE_REFERENCE_QUERIES = {
    "this video", "that video", "this one", "that one", "it", "the same video", "the same one",
}


class LegacyRuleBasedOrchestrator(AgentOrchestrator):
    """Keyword-matching router carried over from the pre-Phase-0 project.

    Behavior is intentionally unchanged from the original `app.py` so
    that existing voice commands ("open YouTube and play ...", "email
    john at gmail dot com type ...") keep working while the real agent
    architecture (tool registry + LLM-driven planning) is built out in
    Phases 3-9.
    """

    def __init__(self, *, tool_registry: ToolRegistry | None = None) -> None:
        # Optional so existing call sites/tests that don't care about
        # YouTube still work; real wiring happens in the app factory.
        self._tool_registry = tool_registry

    def handle(self, command: str, *, context: "AgentRequestContext | None" = None) -> AgentActionResult:
        # `context` carries the connected Google account's access token (if
        # any), used by the real-tool-backed branches added in Tasks 60/61
        # below. This router still has no LLM and no conversation memory,
        # so it can only act on what's literally in the command text - bare
        # references like "this video"/"reply to it" still need the AI.
        cmd = command.strip().lower()

        if any(keyword in cmd for keyword in _YOUTUBE_TRIGGER_WORDS):
            return self._handle_youtube(cmd, context)

        if any(keyword in cmd for keyword in _GMAIL_TRIGGER_WORDS):
            return self._handle_gmail(cmd, context)

        # Task 15 fix: see `_GMAIL_BARE_ACTION_WORDS`/`_GMAIL_DRAFT_TRIGGER_WORDS`
        # above - "Archive this."/"Star this."/"Mark this as read."/"Reply
        # all."/"Show my drafts." etc. are Gmail commands even though they
        # don't contain a `_GMAIL_TRIGGER_WORDS` match.
        if (
            any(keyword in cmd for keyword in _GMAIL_BARE_ACTION_WORDS)
            or any(keyword in cmd for keyword in _GMAIL_DRAFT_TRIGGER_WORDS)
            or re.search(r"\bmark\b.*\b(this|that|it)\b.*\b(read|unread)\b", cmd)
        ):
            return self._handle_gmail(cmd, context)

        return AgentActionResult(
            success=False,
            message="Command not understood. Try a YouTube or Gmail related request.",
            action_type="unknown",
        )

    def _handle_youtube(self, cmd: str, context: "AgentRequestContext | None" = None) -> AgentActionResult:
        """Route a YouTube-flavored command (Task 61: natural-language
        YouTube commands for the no-LLM fallback path).

        "Search YouTube for ..." keeps working exactly as before. "Show my
        channel" is wired to the real `youtube.list_channels` tool (no
        reference resolution needed - "my channel" is unambiguous even
        with zero conversation memory). "Play this video" / "this channel"
        and "create a playlist" both need something this stateless router
        genuinely doesn't have - session memory for the former, the
        confirmation-gate the real planner enforces for the latter (Task
        63/planner's `requires_confirmation` handling) - so they get a
        clear "needs the AI assistant" message instead of a wrong guess.
        """
        access_token = context.access_token if context else None

        if re.search(r"\b(create|make|start)\b.*\bplaylist\b", cmd):
            return AgentActionResult(
                success=False,
                message=(
                    "Creating a playlist needs the local AI model (it asks you to confirm before creating "
                    "anything) - please start Ollama and try again."
                ),
                action_type="youtube_confirmation_unavailable",
            )

        if re.search(r"\bmy\s+(youtube\s+)?channel\b", cmd) or "channel info" in cmd:
            return self._show_my_channel(access_token)

        if re.search(r"\b(this|that)\s+channel\b", cmd):
            return AgentActionResult(
                success=False,
                message="I can only look up your own connected channel right now - try \"show my channel\".",
                action_type="youtube_unavailable",
            )

        query = cmd
        for pattern in _YOUTUBE_STRIP_PATTERNS:
            query = query.replace(pattern, "")
        query = query.strip()

        if not query:
            return AgentActionResult(
                success=False,
                message="Tell me what to search for on YouTube.",
                action_type="youtube_play",
            )

        if query in _BARE_REFERENCE_QUERIES:
            return AgentActionResult(
                success=False,
                message=(
                    "I can't tell which video you mean without the AI assistant - please start Ollama, "
                    "or tell me the title to search for."
                ),
                action_type="youtube_reference_unavailable",
            )

        if self._tool_registry is None:
            return AgentActionResult(
                success=False,
                message="YouTube is not available right now.",
                action_type="youtube_play",
            )

        try:
            tool = self._tool_registry.get("youtube.search")
            result = tool.execute({"query": query, "max_results": 1})
        except NovaError as exc:
            return AgentActionResult(
                success=False,
                message=f"Couldn't search YouTube for '{query}': {exc.message}",
                action_type="youtube_play",
            )

        results = result.data.get("results", []) if result.success else []
        if not results:
            return AgentActionResult(
                success=False,
                message=f"Couldn't find a video for '{query}'.",
                action_type="youtube_play",
            )

        video = results[0]
        video_id = video["video_id"]
        card = build_video_card(video)
        # `url` is now a plain watch-page link for legacy clients/the
        # "Open on YouTube" action, not an autoplay embed. Real playback
        # (Phase 4) goes through the official YouTube IFrame Player API
        # (`static/js/youtube-player.js`) driven by `data`/`actions` below,
        # which correctly reports state instead of assuming autoplay works.
        url = video.get("url") or f"https://www.youtube.com/watch?v={video_id}"
        return AgentActionResult(
            success=True,
            message=f"Found {video['title']}",
            url=url,
            action_type="youtube_play",
            data={"video_id": video_id, "query": query, "video": video, "card": card},
            actions=card["actions"],
        )

    def _show_my_channel(self, access_token: str | None) -> AgentActionResult:
        if not access_token:
            return AgentActionResult(
                success=False,
                message="Connect a Google account first (Settings > Connect Gmail) to see your channel.",
                action_type="youtube_unavailable",
            )
        if self._tool_registry is None:
            return AgentActionResult(success=False, message="YouTube is not available right now.", action_type="youtube_unavailable")
        try:
            tool = self._tool_registry.get("youtube.list_channels")
            result = tool.execute({"access_token": access_token})
        except NovaError as exc:
            return AgentActionResult(success=False, message=exc.message, action_type="youtube_unavailable")

        channel = result.data.get("channel") if result.success else None
        if not channel:
            return AgentActionResult(success=False, message="Couldn't find a channel on this account.", action_type="youtube_unavailable")
        subs = channel.get("subscriber_count")
        subs_text = f" ({subs:,} subscribers)" if isinstance(subs, int) else ""
        return AgentActionResult(
            success=True,
            message=f"Channel: {channel.get('title', 'Unknown channel')}{subs_text}.",
            action_type="youtube_channel",
            data={"channel": channel},
        )


    def _handle_gmail(self, cmd: str, context: "AgentRequestContext | None" = None) -> AgentActionResult:
        """Route a Gmail-flavored command (Task 60: natural-language Gmail
        commands for the no-LLM fallback path).

        Read-only/listing commands ("show my latest emails", "find emails
        from John", "read my latest email") now hit the real `gmail.*`
        tools instead of the old keyword-parser-into-a-mailto-link hack,
        so they work even when the LLM planner (`PlannerOrchestrator`) has
        degraded to this router because Ollama isn't reachable (Task 58).
        Composing/drafting ("draft an email to John ... type ...") still
        reports a plain "Drafting email to ..." message when no Google
        account is connected (or no recipient could be parsed) - Task 16
        removed the `mail.google.com` compose-link fallback this used to
        return, since nothing ever read it (Tasks 06/07 already routed
        every gmail-flavored result away from the external-open fallback
        before it was reached). With a connected account it instead
        renders as a `gmail_draft` card in the internal workspace (Task
        12), matching every other Gmail result type.
        """
        access_token = context.access_token if context else None

        if re.search(r"\bread\b.*\b(latest|last|newest|most recent)\b.*\bemail", cmd):
            return self._read_latest_email(access_token)

        match = re.search(r"\bemails?\s+from\s+(.+)", cmd)
        if match:
            sender = match.group(1).strip().rstrip(" .!?")
            if sender:
                return self._search_emails(f"from:{sender}", access_token, description=f"from {sender}")

        # Task 11: "search gmail for project meeting" / "search my emails
        # for ..." - a plain keyword search, distinct from the
        # sender-scoped "emails from X" above. Routes to the same real
        # `gmail.search` tool (via `_search_emails`) with the query text
        # itself, not a fabricated "from:" filter. Before this, a command
        # like this matched none of the existing branches and fell to
        # `_compose_or_draft`, which (like the "open gmail"/"open this
        # email" bugs fixed in Tasks 09/10) had no real recipient to parse
        # and would have produced a garbage draft address instead of a
        # search.
        match = re.search(r"\bsearch\b.*\b(?:gmail|emails?|mail)\b.*?\bfor\s+(.+)", cmd)
        if match:
            query = match.group(1).strip().rstrip(" .!?")
            if query:
                return self._search_emails(query, access_token, description=f'matching "{query}"')

        if re.search(r"\b(show|list|check|open)\b.*\b(inbox|emails?)\b", cmd):
            return self._list_latest_emails(access_token)

        # Task 09: a bare "Open Gmail" (no "inbox"/"emails" - that's the
        # branch above, and no specific message reference - that's Task
        # 10's "open this email") should just open the internal Gmail
        # workspace panel, not fall through to `_compose_or_draft` below.
        # Before this, "open gmail" had no recipient/body to parse and so
        # produced an actual (if harmless) compose draft - the wrong
        # intent for what is clearly a "show me Gmail" command.
        if re.search(r"\bopen\b", cmd) and re.search(r"\bgmail\b", cmd):
            return AgentActionResult(
                success=True,
                message="Here's your Gmail workspace.",
                action_type="gmail_workspace_open",
                data={},
            )

        if "reply" in cmd:
            return AgentActionResult(
                success=False,
                message=(
                    "Writing a reply needs the local AI model to read the email and draft the text - "
                    "please start Ollama and try again."
                ),
                action_type="gmail_reply_unavailable",
            )

        # Task 15 fix: "Forward this email." reached here with no branch
        # to catch it (unlike "reply", which the check just above already
        # handled) and fell through to `_compose_or_draft`, which has no
        # real recipient/body to parse out of "forward this email" and
        # would have silently created a garbage draft. Forwarding needs
        # the same conversation memory (which email is "this email"?) the
        # AI-backed planner has and this stateless router doesn't.
        if "forward" in cmd:
            return AgentActionResult(
                success=False,
                message=(
                    "Forwarding needs the local AI model to know which email you mean - "
                    "please start Ollama and try again."
                ),
                action_type="gmail_forward_unavailable",
            )

        # Task 15 fix: bare actions on an already-open email ("Archive
        # this.", "Delete this.", "Star this.", "Unstar this.", "Mark this
        # as read.") all reference session state this router doesn't keep
        # (see class docstring) - same category as "reply"/"forward"
        # above, so they get the same "needs the AI assistant" treatment
        # instead of being misparsed as a compose command by
        # `_compose_or_draft` below.
        if any(
            keyword in cmd for keyword in ("archive", "trash", "delete", "untrash", "restore", "star", "unstar")
        ) or re.search(r"\bmark\b.*\b(this|that|it)\b.*\b(read|unread)\b", cmd):
            return AgentActionResult(
                success=False,
                message=(
                    "That needs the local AI model to know which email you mean - "
                    "please start Ollama and try again."
                ),
                action_type="gmail_action_unavailable",
            )

        # Task 15 fix: "Show my drafts." / "List my drafts." is unambiguous
        # (no "this"/"that" reference to resolve, same reasoning as "show my
        # channel" in `_handle_youtube`) so it can hit the real
        # `gmail.draft.list` tool directly. "Edit this draft." / "Send this
        # draft." reference a specific already-open draft this stateless
        # router has no memory of, so - same pattern as "reply to this
        # email" above - they get a "needs the AI assistant" message
        # instead of being misrouted into `_compose_or_draft` as a brand
        # new blank draft.
        if "draft" in cmd:
            if re.search(r"\b(this|that|it)\b", cmd):
                return AgentActionResult(
                    success=False,
                    message=(
                        "I can't tell which draft you mean without the AI assistant - please start Ollama, "
                        "or ask to \"show my drafts\"."
                    ),
                    action_type="gmail_draft_reference_unavailable",
                )
            if re.search(r"\b(show|list|open|check)\b.*\bdrafts?\b", cmd):
                return self._list_drafts(access_token)

        # Task 10: "open this email" / "read that message" reference an
        # already-shown email this stateless router has no memory of (see
        # class docstring and the identical "this video"/"this channel"
        # carve-out in `_handle_youtube` above) - without this check the
        # command fell through to `_compose_or_draft`, which had no
        # recipient/body to parse either and quietly turned "open this
        # email" into a draft addressed to "openthisemail@gmail.com". A
        # clear "needs the AI assistant" message is correct here, same as
        # the YouTube case, rather than guessing or fabricating a message.
        if re.search(r"\b(open|read|show)\b.*\b(this|that)\b.*\b(email|message|mail)\b", cmd):
            return AgentActionResult(
                success=False,
                message=(
                    "I can't tell which email you mean without the AI assistant - please start Ollama, "
                    "or ask for \"the latest email\" / search by sender."
                ),
                action_type="gmail_reference_unavailable",
            )

        return self._compose_or_draft(cmd, access_token)

    def _run_gmail_tool(self, tool_name: str, arguments: dict[str, Any]) -> AgentActionResult | dict[str, Any]:
        """Execute a registered `gmail.*` tool, translating failures into an
        `AgentActionResult`. Returns the raw `.data` dict on success so
        callers can build their own message/card around it.
        """
        if self._tool_registry is None:
            return AgentActionResult(success=False, message="Gmail is not available right now.", action_type="gmail_unavailable")
        try:
            tool = self._tool_registry.get(tool_name)
            result = tool.execute(arguments)
        except NovaError as exc:
            return AgentActionResult(success=False, message=exc.message, action_type="gmail_unavailable")
        if not result.success:
            return AgentActionResult(success=False, message=result.error or "That Gmail request failed.", action_type="gmail_unavailable")
        return result.data

    def _require_connected_account(self, access_token: str | None) -> AgentActionResult | None:
        if access_token:
            return None
        return AgentActionResult(
            success=False,
            message="Connect a Google account first (Settings > Connect Gmail) to check your inbox.",
            action_type="gmail_unavailable",
        )

    def _list_latest_emails(self, access_token: str | None) -> AgentActionResult:
        missing = self._require_connected_account(access_token)
        if missing is not None:
            return missing
        outcome = self._run_gmail_tool("gmail.list_messages", {"access_token": access_token, "max_results": 5})
        if isinstance(outcome, AgentActionResult):
            return outcome
        messages = outcome.get("messages") or []
        cards = build_gmail_message_cards(messages)
        message = "Your inbox is empty." if not messages else f"Here are your {len(messages)} most recent emails."
        return AgentActionResult(
            success=True, message=message, action_type="gmail_message_list", data={"messages": messages, "cards": cards}
        )

    def _list_drafts(self, access_token: str | None) -> AgentActionResult:
        missing = self._require_connected_account(access_token)
        if missing is not None:
            return missing
        outcome = self._run_gmail_tool("gmail.draft.list", {"access_token": access_token, "max_results": 10})
        if isinstance(outcome, AgentActionResult):
            return outcome
        drafts = outcome.get("drafts") or []
        cards = build_gmail_draft_cards(drafts)
        message = "You have no saved drafts." if not drafts else f"Here are your {len(drafts)} draft(s)."
        return AgentActionResult(
            success=True, message=message, action_type="gmail_draft_list", data={"drafts": drafts, "cards": cards}
        )

    def _search_emails(self, query: str, access_token: str | None, *, description: str) -> AgentActionResult:
        missing = self._require_connected_account(access_token)
        if missing is not None:
            return missing
        outcome = self._run_gmail_tool("gmail.search", {"access_token": access_token, "query": query, "max_results": 5})
        if isinstance(outcome, AgentActionResult):
            return outcome
        messages = outcome.get("messages") or []
        cards = build_gmail_message_cards(messages)
        message = f"No emails found {description}." if not messages else f"Found {len(messages)} email(s) {description}."
        return AgentActionResult(
            success=True, message=message, action_type="gmail_message_list", data={"messages": messages, "cards": cards}
        )

    def _read_latest_email(self, access_token: str | None) -> AgentActionResult:
        missing = self._require_connected_account(access_token)
        if missing is not None:
            return missing
        outcome = self._run_gmail_tool("gmail.list_messages", {"access_token": access_token, "max_results": 1})
        if isinstance(outcome, AgentActionResult):
            return outcome
        messages = outcome.get("messages") or []
        if not messages:
            return AgentActionResult(success=True, message="Your inbox is empty.", action_type="gmail_message_list", data={"messages": []})

        outcome = self._run_gmail_tool(
            "gmail.get_message", {"access_token": access_token, "message_id": messages[0]["message_id"]}
        )
        if isinstance(outcome, AgentActionResult):
            return outcome
        message = outcome.get("message")
        if not message:
            return AgentActionResult(success=False, message="Couldn't open that email.", action_type="gmail_unavailable")
        card = build_gmail_message_cards([message])[0]
        return AgentActionResult(
            success=True,
            message=f"Here's your latest email: {message.get('subject', '(no subject)')}.",
            action_type="gmail_message",
            data={"message": message, "card": card},
            actions=card["actions"],
        )

    def _compose_or_draft(self, cmd: str, access_token: str | None) -> AgentActionResult:
        # Strips leading filler so both "email john at gmail dot com type
        # ..." (original phrasing) and "draft/compose an email to john ..."
        # (Task 60's "Draft an email to John" example) land on the same
        # recipient text afterwards.
        clean_cmd = re.sub(
            r"^(please\s+)?(draft|compose|write|send|open)?\s*(an?\s+)?(new\s+)?(gmail|email|mail|message)\s*(to\s+)?",
            "",
            cmd,
        ).strip()
        clean_cmd = re.sub(r"\b(com(and|mand)?)\b", "com", clean_cmd)

        parts = re.split(r"\b(type|write|saying|message|content|with body)\b", clean_cmd)
        recipient_part = parts[0].strip()
        recipient_part = re.sub(r"^(update\s+to|to|send\s+to|and\s+update\s+to)\s*", "", recipient_part).strip()

        body = parts[-1].strip() if len(parts) > 1 else ""

        to = ""
        if recipient_part:
            candidate = recipient_part.replace(" at ", "@").replace(" dot ", ".").replace(" ", "")
            candidate = re.sub(r"[^a-zA-Z0-9@._%-]", "", candidate)
            to = candidate if "@" in candidate else f"{candidate}@gmail.com"

        # With a connected Google account, actually create the draft via
        # the real Gmail API (nothing is sent - see gmail.draft.create's
        # `requires_confirmation = False`) instead of only opening Gmail's
        # own compose window.
        if access_token and to:
            outcome = self._run_gmail_tool(
                "gmail.draft.create", {"access_token": access_token, "to": [to], "subject": "", "body_text": body}
            )
            if not isinstance(outcome, AgentActionResult):
                draft = outcome.get("draft")
                if draft:
                    # Task 12: include the `card` key (missing before), the
                    # one thing the frontend's Gmail-branch card renderer
                    # (`app/templates/index.html`'s
                    # `data.data && data.data.card` check) actually looks
                    # for - without it this successful draft-create had no
                    # visible result at all beyond a status line, even
                    # though the draft really was saved.
                    card = build_gmail_draft_card(draft)
                    return AgentActionResult(
                        success=True,
                        message=f"Saved a draft to {to}.",
                        action_type="gmail_draft",
                        data={"draft": draft, "card": card},
                        actions=card["actions"],
                    )

        # Task 16: this used to fall back to a `mail.google.com/...
        # ?view=cm&fs=1` compose-window deep link (`url=...`) for
        # whichever case skipped the internal-draft branch above (no
        # access token yet, or no recipient the regex could parse). That
        # was the last obsolete Gmail external-opening code path in the
        # fallback router - no `url` is returned now, so there is nothing
        # here for a client to `window.open()`/navigate to; the message
        # still reports what Nova understood, same as before.
        return AgentActionResult(
            success=True,
            message=f"Drafting email to {to}" if to else "Drafting email",
            action_type="gmail_compose",
            data={"to": to, "body": body},
        )
