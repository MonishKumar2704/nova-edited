"""
Real agent orchestrator (Phase 9: REAL AGENT ORCHESTRATOR, master spec
section 9).

Replaces the `if "youtube" in command: ... elif "gmail" in command: ...`
style routing of `LegacyRuleBasedOrchestrator` with genuine intent
interpretation, tool discovery, and (bounded) multi-step planning, built
entirely on top of scaffolding shipped in earlier phases:

  * `app.llm.base.LLMProvider` (Phase 8) - provider-independent
    generation + tool-calling (Gemini/Ollama; free-first, master spec
    section 15).
  * `app.tools.registry.ToolRegistry` (Phase 0) - dynamic tool discovery.
    `PlannerOrchestrator` never imports a concrete tool; it hands the
    registry's `describe_all()` output to the LLM and looks up whatever
    name comes back (master spec section 7: no giant if/elif router).
  * `Tool.requires_confirmation` (Phase 3+) - the confirmation gate.
  * `app.core.errors` classified errors - tool/LLM failures never crash
    the loop, they get fed back to the model or surfaced to the user.
  * `app.agent.state` - per-session structured memory so "play the third
    result" / "reply to that" / "send it" resolve against real prior
    turns instead of the model hallucinating IDs (master spec: "the LLM
    must not fabricate video IDs / emails / URLs / statistics").

Flow (`handle()`):
  1. If this call is *answering* a pending confirmation
     (`AgentRequestContext.confirm is not None`), resolve it directly -
     no LLM round trip needed.
  2. Otherwise run a bounded plan/act loop: ask the LLM what to do
     (given the tool specs + session state + short history), execute at
     most one tool call per round, feed the (compact) result back, and
     repeat until the model returns a final text answer or the
     iteration budget (`Config.agent_max_tool_iterations`) runs out.
  3. A tool with `requires_confirmation = True` is never executed
     inline - the loop stops, stashes the call in
     `AgentState.pending_confirmation`, and returns a
     `confirmation_required` result.
  4. If no LLM provider is configured/reachable - `NullProvider`
     (`LLMError`), or a configured Ollama that isn't running/timing
     out/overloaded (`NetworkError` / `TimeoutErrorNova` /
     `RateLimitError`) - degrade to the wrapped
     `LegacyRuleBasedOrchestrator` rather than failing the whole agent
     or crashing (Task 58): free-first (section 15) means "the local AI
     isn't reachable" must still leave Nova usable for direct
     YouTube/Gmail commands, with a plain-English "start Ollama" hint
     for the requests that actually needed it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agent.context import AgentRequestContext
from app.agent.orchestrator import AgentActionResult, AgentOrchestrator, LegacyRuleBasedOrchestrator
from app.agent.state import AgentState, AgentStateStore, PendingConfirmation
from app.core.errors import LLMError, NetworkError, NovaError, RateLimitError, TimeoutErrorNova, ToolError, ValidationError
from app.llm.base import LLMMessage, LLMProvider
from app.services.dynamic_ui import (
    build_gmail_draft_card,
    build_gmail_message_card,
    build_gmail_message_cards,
    build_gmail_thread_card,
    build_playlist_card,
    build_playlist_cards,
    build_playlist_item_card,
    build_playlist_item_cards,
    build_video_card,
)
from app.tools.base import Tool, ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are Nova, a voice-and-text assistant that controls YouTube and Gmail \
on the user's behalf through a fixed set of tools. Follow these rules strictly:

1. Only use information returned by tool calls. Never invent video IDs, titles, channel \
names, URLs, view counts, email addresses, subjects, or message content - if you don't have \
a fact from a tool result or the session context below, call a tool to get it or ask the user.
2. Prefer using the "Session context" below to resolve references such as "the third result", \
"that video", "this email", "reply to it", or "send it" instead of asking the user to repeat \
themselves, as long as the reference is unambiguous.
3. If a request is missing information required to proceed (e.g. no recipient for an email, \
no query for a search) and it cannot be inferred from context, do NOT guess - respond with a \
short plain-text clarifying question and do not call a tool.
4. Call at most one tool per turn. After a tool result is given back to you, decide whether \
you are done (respond with plain text summarizing the outcome for the user) or need another tool call.
5. Some tools require user confirmation before they run (e.g. sending an email, deleting a \
playlist) - the system enforces this automatically; you do not need to ask for confirmation \
yourself, just select the tool as normal.
6. Keep replies conversational and concise - the user is likely listening to a voice response.
7. For "write/draft an email about X to Y" where Y is an actual email address, call `ai.email.draft` \
directly (it generates the text AND saves the draft in one step) rather than calling `ai.email.generate` \
first - don't chain the two. Use `ai.email.generate` only when the user wants to see generated text \
without saving anything (e.g. "what would an email like that even say").
8. For an existing email the user pastes in, use `ai.email.grammar_correct` only when they specifically \
ask to fix spelling/grammar/punctuation mistakes; use `ai.email.rewrite` for anything broader (tone, \
length, style, or a general "improve this").
9. For "reply to this email" / "draft a reply" against an email already open in the session (see \
"Session context" below for `message_id`), call `ai.email.suggest_reply` - it reads the email, writes a \
reply with the local AI model, and saves it as a draft in one step. Only use `gmail.reply` directly when \
the user has already told you exactly what the reply should say.
10. For "play this video" / "play it again" / "open that video" referring to a video already in the \
Session context (not a new search), call `youtube.get_video` with its known `video_id` so the reply comes \
back with a working play card - don't just describe the video in plain text without calling the tool, or \
the user has nothing to actually press play on.
11. Not every message is a Gmail or YouTube request. General questions, small talk, or anything you can \
already answer from your own knowledge (e.g. "what's the capital of France", "explain how photosynthesis \
works", "tell me a joke") should get a direct plain-text answer - do NOT call a Gmail or YouTube tool just \
because one is available if the user didn't actually ask for an email or video action."""

# Tool-name prefixes/exact names that get a friendlier, spec-aware
# confirmation description than the generic "Run <tool> with <args>"
# fallback. Kept data-driven (not a growing if/elif chain) so adding a
# new sensitive tool just means adding one entry here.
#
# `gmail.draft.send` is deliberately NOT in this dict: unlike the other
# entries, its arguments are just a `draft_id` (no subject/body), so a
# static lambda can't preview what's actually about to be sent - and
# `gmail.draft.send` is the one send tool every AI-generated email
# (`ai.email.draft`, `ai.email.suggest_reply`) goes through. Its
# description is built in `_describe_call` from `AgentState.current_draft`
# instead, so the user sees the real subject/body before confirming
# (Task 59: request -> AI generation -> draft -> *preview* -> confirm -> send).
_CONFIRMATION_DESCRIPTIONS = {
    "gmail.send": lambda a: f"Send an email to {', '.join(a.get('to') or [])} with subject '{a.get('subject', '')}'.",
    "gmail.reply": lambda a: "Send a reply to this email.",
    "gmail.reply_all": lambda a: "Send a reply-all to this email.",
    "gmail.forward": lambda a: f"Forward this email to {', '.join(a.get('to') or [])}.",
    "gmail.archive": lambda a: "Archive this email.",
    "gmail.trash": lambda a: "Delete this email.",
    "gmail.draft.delete": lambda a: "Permanently delete this draft.",
    "youtube.playlist.delete": lambda a: "Delete this playlist.",
    "youtube.playlist.add_video": lambda a: "Add this video to the playlist.",
    "youtube.playlist.remove_video": lambda a: "Remove this video from the playlist.",
    "youtube.video.update": lambda a: (
        f"Update the video's title to '{a['title']}'." if a.get("title") else "Update this video's details."
    ),
    "youtube.video.delete": lambda a: "Delete this video.",
    "youtube.video.rate": lambda a: (
        {"like": "Like this video.", "dislike": "Dislike this video.", "none": "Clear your rating on this video."}.get(
            a.get("rating"), "Rate this video."
        )
    ),
}

_DRAFT_BODY_PREVIEW_CHARS = 160

_MAX_TOOL_RESULT_CHARS = 4000

# Errors that mean "the configured LLM backend (free local Ollama by
# default) isn't usable right now" - whether that's nothing configured
# (`LLMError`) or Ollama specifically being down/slow/overloaded
# (`NetworkError` / `TimeoutErrorNova` / `RateLimitError`). All of these
# should degrade to the legacy keyword router (Task 58) instead of
# bubbling up as a raw 5xx - a local model not running is an expected,
# non-fatal condition, not an application bug.
_LLM_UNAVAILABLE_ERRORS: tuple[type[NovaError], ...] = (LLMError, NetworkError, TimeoutErrorNova, RateLimitError)

_OLLAMA_UNAVAILABLE_MESSAGE = (
    "I can't reach the local AI model right now - Ollama may not be running. "
    "Start it (e.g. `ollama serve`) and try again. In the meantime I can still "
    "handle direct YouTube or Gmail commands."
)


def _compact_json(data: Any) -> str:
    try:
        text = json.dumps(data, default=str)
    except (TypeError, ValueError):
        text = str(data)
    if len(text) > _MAX_TOOL_RESULT_CHARS:
        text = text[:_MAX_TOOL_RESULT_CHARS] + "...(truncated)"
    return text


def _draft_send_description(arguments: dict[str, Any], state: "AgentState | None") -> str | None:
    """Build a content preview for confirming `gmail.draft.send` (Task 59).

    Only trusts `AgentState.current_draft` when its `draft_id` matches the
    one actually being sent - if it doesn't match (or nothing is tracked),
    return None so the caller falls back to the generic description rather
    than showing the wrong email's content.
    """
    if state is None:
        return None
    draft = state.current_draft
    if not draft or draft.get("draft_id") != arguments.get("draft_id"):
        return None

    message = draft.get("message") or {}
    to = message.get("to") or ""
    subject = message.get("subject") or "(no subject)"
    body = (message.get("body_text") or "").strip().replace("\n", " ")
    if len(body) > _DRAFT_BODY_PREVIEW_CHARS:
        body = body[:_DRAFT_BODY_PREVIEW_CHARS].rstrip() + "..."

    to_part = f" to {to}" if to else ""
    body_part = f': "{body}"' if body else ""
    return f"Send the draft{to_part} - Subject: '{subject}'{body_part}."


def _describe_call(tool: Tool, arguments: dict[str, Any], state: "AgentState | None" = None) -> str:
    if tool.name == "gmail.draft.send":
        preview = _draft_send_description(arguments, state)
        if preview is not None:
            return preview
        return "Send the saved draft."

    builder = _CONFIRMATION_DESCRIPTIONS.get(tool.name)
    if builder is not None:
        try:
            return builder(arguments)
        except Exception:  # noqa: BLE001 - fall through to the generic description below
            pass
    return f"Run {tool.name} ({_compact_json(arguments)})."


def _action(action_id: str, label: str) -> dict[str, Any]:
    return {"id": action_id, "label": label, "requires_confirmation": False}


class PlannerOrchestrator(AgentOrchestrator):
    """LLM-driven planner/tool-selecting orchestrator (Phase 9)."""

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        state_store: AgentStateStore,
        legacy_fallback: LegacyRuleBasedOrchestrator,
        max_tool_iterations: int = 5,
        llm_timeout: float = 20.0,
    ) -> None:
        self._llm = llm_provider
        self._registry = tool_registry
        self._state_store = state_store
        self._legacy = legacy_fallback
        self._max_iterations = max(1, max_tool_iterations)
        self._llm_timeout = llm_timeout

    # -- AgentOrchestrator interface -------------------------------------------------

    def handle(self, command: str, *, context: AgentRequestContext | None = None) -> AgentActionResult:
        ctx = context or AgentRequestContext(session_id="anonymous")
        state = self._state_store.get_or_create(ctx.session_id)

        if ctx.confirm is not None:
            return self._resolve_confirmation(state, ctx)

        clean_command = (command or "").strip()
        if not clean_command:
            return AgentActionResult(
                success=False, message="Tell me what you'd like to do.", action_type="agent_reply"
            )

        try:
            return self._run_plan_loop(clean_command, state, ctx)
        except _LLM_UNAVAILABLE_ERRORS as exc:
            logger.info("AI planner unavailable (%s); falling back to rule-based routing.", exc)
            result = self._legacy.handle(clean_command, context=ctx)
            if not result.success and result.action_type == "unknown":
                # The legacy router doesn't understand this command either
                # (it's not a plain YouTube/Gmail keyword match) - most
                # likely it needed the AI itself, so say so plainly instead
                # of the generic "command not understood".
                result.message = _OLLAMA_UNAVAILABLE_MESSAGE
            return result

    # -- confirmation handling -------------------------------------------------

    def _resolve_confirmation(self, state: AgentState, ctx: AgentRequestContext) -> AgentActionResult:
        with state.lock:
            pending = state.pending_confirmation
            state.pending_confirmation = None

        if pending is None:
            return AgentActionResult(
                success=False, message="There's nothing pending to confirm.", action_type="agent_reply"
            )

        if ctx.confirm is False:
            state.record_turn(f"[declined] {pending.description}", "Okay, cancelled.")
            return AgentActionResult(success=True, message="Okay, cancelled.", action_type="agent_reply")

        try:
            tool = self._registry.get(pending.tool_name)
        except ToolError as exc:
            return AgentActionResult(success=False, message=str(exc.message), action_type="tool_error")

        arguments = dict(pending.arguments)
        try:
            self._inject_access_token(tool, arguments, ctx)
        except ValidationError as exc:
            return AgentActionResult(success=False, message=exc.message, action_type="agent_reply")

        try:
            result = tool.execute(arguments)
        except NovaError as exc:
            message = f"Couldn't complete that: {exc.message}"
            state.record_turn(f"[confirmed] {pending.description}", message)
            return AgentActionResult(
                success=False, message=message, action_type="tool_error", data={"error_code": exc.code}
            )

        self._update_state_from_tool_result(state, tool.name, result)
        action_result = self._build_action_result(tool.name, result)
        state.record_turn(f"[confirmed] {pending.description}", action_result.message)
        return action_result

    # -- plan/act loop -------------------------------------------------

    def _run_plan_loop(self, command: str, state: AgentState, ctx: AgentRequestContext) -> AgentActionResult:
        messages = self._build_messages(command, state)
        tool_specs = self._registry.describe_all()

        for _ in range(self._max_iterations):
            response = self._llm.generate(messages, tools=tool_specs, timeout=self._llm_timeout)

            if not response.tool_calls:
                text = response.text.strip() or "Done."
                state.record_turn(command, text)
                return AgentActionResult(success=True, message=text, action_type="agent_reply")

            call = response.tool_calls[0]
            messages.append(LLMMessage(role="assistant", content=response.text or f"(calling {call.name})"))

            try:
                tool = self._registry.get(call.name)
            except ToolError:
                messages.append(
                    LLMMessage(
                        role="user",
                        content=f"[tool_result for {call.name}] error: unknown tool. "
                        "Pick one of the tools listed in your instructions.",
                    )
                )
                continue

            if tool.requires_confirmation:
                description = _describe_call(tool, call.arguments, state)
                with state.lock:
                    state.pending_confirmation = PendingConfirmation(
                        tool_name=tool.name, arguments=dict(call.arguments), description=description
                    )
                message = f"{description} Shall I go ahead?"
                state.record_turn(command, message)
                return AgentActionResult(
                    success=True,
                    message=message,
                    action_type="confirmation_required",
                    data={"tool": tool.name, "arguments": call.arguments},
                    actions=[_action("confirm", "Confirm"), _action("cancel", "Cancel")],
                )

            arguments = dict(call.arguments)
            try:
                self._inject_access_token(tool, arguments, ctx)
            except ValidationError as exc:
                state.record_turn(command, exc.message)
                return AgentActionResult(success=False, message=exc.message, action_type="agent_reply")

            try:
                tool_result = tool.execute(arguments)
            except NovaError as exc:
                messages.append(
                    LLMMessage(role="user", content=f"[tool_result for {call.name}] error ({exc.code}): {exc.message}")
                )
                continue

            self._update_state_from_tool_result(state, tool.name, tool_result)

            if not tool_result.success:
                messages.append(
                    LLMMessage(
                        role="user",
                        content=f"[tool_result for {call.name}] failed: {tool_result.error or 'unknown error'}",
                    )
                )
                continue

            messages.append(
                LLMMessage(role="user", content=f"[tool_result for {call.name}] {_compact_json(tool_result.data)}")
            )

        message = "That's taking more steps than expected - could you narrow down what you'd like me to do?"
        state.record_turn(command, message)
        return AgentActionResult(success=False, message=message, action_type="agent_incomplete")

    # -- helpers -------------------------------------------------

    def _build_messages(self, command: str, state: AgentState) -> list[LLMMessage]:
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="system", content=f"Session context:\n{state.summarize_for_prompt()}"),
        ]
        for turn in state.conversation_history:
            messages.append(LLMMessage(role=turn["role"], content=turn["content"]))
        messages.append(LLMMessage(role="user", content=command))
        return messages

    def _inject_access_token(self, tool: Tool, arguments: dict[str, Any], ctx: AgentRequestContext) -> None:
        if "access_token" not in tool.input_schema or arguments.get("access_token"):
            return
        if not ctx.access_token:
            raise ValidationError(
                f"'{tool.name}' needs a connected Google account. Connect one via "
                "/api/v1/auth/google/connect and try again."
            )
        arguments["access_token"] = ctx.access_token

    def _build_action_result(self, tool_name: str, result: ToolResult) -> AgentActionResult:
        """Turn a successful `ToolResult` into a user-facing `AgentActionResult`.

        Reuses the exact same card builders the direct REST routes use
        (`app.services.dynamic_ui`) so a video/playlist/message/draft the
        agent surfaces renders identically to one surfaced by a direct
        `/api/v1/...` call - one dynamic-UI contract, one frontend
        renderer, regardless of which path produced the data (master
        spec section 10).
        """
        data = result.data

        if tool_name in ("youtube.search", "youtube.get_video") and data.get("results"):
            video = data["results"][0]
            card = build_video_card(video)
            return AgentActionResult(
                success=True,
                message=f"Found {video.get('title', 'a video')}.",
                url=video.get("url"),
                action_type="youtube_play",
                data={"video": video, "card": card},
                actions=card["actions"],
            )

        if tool_name == "youtube.playlist.list":
            playlists = data.get("playlists") or []
            if not playlists:
                return AgentActionResult(
                    success=True,
                    message="You don't have any playlists.",
                    action_type="youtube_playlists",
                    data=data,
                )
            cards = build_playlist_cards(playlists)
            noun = "playlist" if len(playlists) == 1 else "playlists"
            return AgentActionResult(
                success=True,
                message=f"Found {len(playlists)} {noun}.",
                action_type="youtube_playlists",
                data={"playlists": playlists, "cards": cards},
                actions=cards[0]["actions"],
            )

        if tool_name == "youtube.playlist.get" and data.get("playlist"):
            playlist = data["playlist"]
            items = data.get("items") or []
            card = build_playlist_card(playlist)
            item_count = f" ({len(items)} video{'s' if len(items) != 1 else ''})" if items else ""
            return AgentActionResult(
                success=True,
                message=f"Playlist '{playlist.get('title', '')}'{item_count}.",
                action_type="youtube_playlist",
                data={"playlist": playlist, "items": items, "card": card, "cards": build_playlist_item_cards(items)},
                actions=card["actions"],
            )

        if tool_name == "youtube.playlist.add_video" and data.get("item"):
            item = data["item"]
            card = build_playlist_item_card(item)
            return AgentActionResult(
                success=True,
                message=f"Added '{item.get('title', 'video')}' to the playlist.",
                action_type="youtube_playlist_item_added",
                data={"item": item, "card": card},
                actions=card["actions"],
            )

        if tool_name == "youtube.playlist.remove_video" and data.get("removed"):
            return AgentActionResult(
                success=True,
                message="Removed from the playlist.",
                action_type="youtube_playlist_item_removed",
                data=data,
            )

        if tool_name == "youtube.playlist.reorder_video" and data.get("item"):
            item = data["item"]
            card = build_playlist_item_card(item)
            position = item.get("position")
            position_text = f" to position {position + 1}" if isinstance(position, int) else ""
            return AgentActionResult(
                success=True,
                message=f"Moved '{item.get('title', 'video')}'{position_text}.",
                action_type="youtube_playlist_item_reordered",
                data={"item": item, "card": card},
                actions=card["actions"],
            )

        if tool_name == "youtube.playlist.delete" and data.get("deleted"):
            return AgentActionResult(
                success=True,
                message="Playlist deleted.",
                action_type="youtube_playlist_deleted",
                data=data,
            )

        if tool_name == "youtube.video.update" and data.get("video"):
            video = data["video"]
            card = build_video_card(video)
            return AgentActionResult(
                success=True,
                message=f"Updated '{video.get('title', 'the video')}'.",
                action_type="youtube_video_updated",
                data={"video": video, "card": card},
                actions=card["actions"],
            )

        if tool_name == "youtube.video.delete" and data.get("deleted"):
            return AgentActionResult(
                success=True,
                message="Video deleted.",
                action_type="youtube_video_deleted",
                data=data,
            )

        if tool_name == "youtube.video.rate" and "rating" in data:
            rating_messages = {
                "like": "Liked the video.",
                "dislike": "Disliked the video.",
                "none": "Cleared your rating on the video.",
            }
            return AgentActionResult(
                success=True,
                message=rating_messages.get(data["rating"], "Rating updated."),
                action_type="youtube_video_rated",
                data=data,
            )

        if tool_name == "youtube.video.get_rating" and "rating" in data:
            rating = data["rating"]
            return AgentActionResult(
                success=True,
                message=f"Your current rating on this video: {rating or 'none'}.",
                action_type="youtube_video_rating",
                data=data,
            )

        if tool_name.startswith("youtube.playlist") and data.get("playlist"):
            playlist = data["playlist"]
            card = build_playlist_card(playlist)
            return AgentActionResult(
                success=True,
                message=f"Playlist '{playlist.get('title', '')}' ready.",
                action_type="youtube_playlist",
                data={"playlist": playlist, "card": card},
                actions=card["actions"],
            )

        if tool_name == "youtube.channel.my_uploads":
            uploads = data.get("uploads") or []
            if not uploads:
                return AgentActionResult(
                    success=True,
                    message="You don't have any uploaded videos.",
                    action_type="youtube_uploads",
                    data=data,
                )
            card = build_playlist_item_card(uploads[0])
            noun = "video" if len(uploads) == 1 else "videos"
            return AgentActionResult(
                success=True,
                message=f"Found {len(uploads)} of your uploaded {noun}.",
                action_type="youtube_uploads",
                data={"uploads": uploads, "card": card},
                actions=card["actions"],
            )

        if tool_name == "youtube.list_channels" and data.get("channel"):
            channel = data["channel"]
            subs = channel.get("subscriber_count")
            subs_text = f" ({subs:,} subscribers)" if isinstance(subs, int) else ""
            return AgentActionResult(
                success=True,
                message=f"Channel: {channel.get('title', 'Unknown channel')}{subs_text}.",
                action_type="youtube_channel",
                data=data,
            )

        if tool_name == "ai.email.generate" and data.get("subject") is not None:
            subject = data["subject"]
            body = data.get("body", "")
            return AgentActionResult(
                success=True,
                message=f"Here's a draft email:\n\nSubject: {subject}\n\n{body}",
                action_type="ai_email_generated",
                data={"subject": subject, "body": body},
            )

        if tool_name == "ai.email.rewrite" and data.get("text") is not None:
            return AgentActionResult(
                success=True,
                message=f"Here's the rewritten email:\n\n{data['text']}",
                action_type="ai_email_rewritten",
                data=data,
            )

        if tool_name == "ai.email.grammar_correct" and data.get("text") is not None:
            return AgentActionResult(
                success=True,
                message=f"Here's the corrected email:\n\n{data['text']}",
                action_type="ai_email_grammar_corrected",
                data=data,
            )

        if tool_name == "ai.email.draft" and data.get("draft"):
            draft = data["draft"]
            card = build_gmail_draft_card(draft)
            subject = data.get("subject", "")
            return AgentActionResult(
                success=True,
                message=f"Saved a draft: '{subject}'.",
                action_type="ai_email_drafted",
                data={"draft": draft, "card": card},
                actions=card["actions"],
            )

        if tool_name == "ai.email.suggest_reply" and data.get("draft"):
            draft = data["draft"]
            card = build_gmail_draft_card(draft)
            return AgentActionResult(
                success=True,
                message=f"Here's a suggested reply, saved as a draft:\n\n{data.get('body', '')}",
                action_type="ai_email_reply_suggested",
                data={"draft": draft, "card": card},
                actions=card["actions"],
            )

        if tool_name in ("gmail.list_messages", "gmail.search") and "messages" in data:
            # Task 60: natural-language listing/search ("show my latest
            # emails", "find emails from John") reuses the exact same
            # `gmail_message` cards the direct `/api/v1/gmail/messages`
            # route and the single-message branch below already render,
            # instead of falling through to the generic "Done." reply.
            messages = data.get("messages") or []
            cards = build_gmail_message_cards(messages)
            if not messages:
                message_text = "No emails match that." if tool_name == "gmail.search" else "Your inbox is empty."
            else:
                message_text = f"Found {len(messages)} email(s)." if tool_name == "gmail.search" else f"Here are your {len(messages)} most recent emails."
            return AgentActionResult(
                success=True,
                message=message_text,
                action_type="gmail_message_list",
                data={"messages": messages, "cards": cards},
            )

        if tool_name in ("gmail.get_message",) and data.get("message"):
            message = data["message"]
            card = build_gmail_message_card(message)
            return AgentActionResult(
                success=True,
                message=f"Here's the email: {message.get('subject', '(no subject)')}.",
                action_type="gmail_message",
                data={"message": message, "card": card},
                actions=card["actions"],
            )

        if tool_name in ("gmail.get_thread",) and data.get("thread"):
            thread = data["thread"]
            card = build_gmail_thread_card(thread)
            return AgentActionResult(
                success=True,
                message="Here's the conversation.",
                action_type="gmail_thread",
                data={"thread": thread, "card": card},
                actions=card["actions"],
            )

        if tool_name.startswith("gmail.draft") and data.get("draft"):
            draft = data["draft"]
            card = build_gmail_draft_card(draft)
            return AgentActionResult(
                success=True,
                message="Draft saved.",
                action_type="gmail_draft",
                data={"draft": draft, "card": card},
                actions=card["actions"],
            )

        if tool_name in ("gmail.send", "gmail.draft.send", "gmail.reply", "gmail.reply_all", "gmail.forward") and data.get(
            "message"
        ):
            return AgentActionResult(success=True, message="Sent.", action_type="gmail_sent", data=data)

        return AgentActionResult(success=result.success, message="Done.", action_type="tool_result", data=data)

    def _update_state_from_tool_result(self, state: AgentState, tool_name: str, result: ToolResult) -> None:
        """Keep `AgentState`'s "what are we talking about" fields current.

        Data-driven on tool name (not a per-tool if/elif scattered through
        the loop) - every tool that returns one of these shapes updates the
        matching state slot automatically, including ones added by future
        tool packages that happen to reuse these output keys.
        """
        if not result.success:
            return
        data = result.data

        with state.lock:
            # Note (Task 15 scope): `current_search_results` is a single,
            # deliberately shared "last list shown" slot reused by both
            # YouTube and Gmail list-producing tools (see the Gmail
            # branches below) so "open the second one" resolves against
            # whatever list was shown most recently, regardless of which
            # feature produced it. That means a YouTube search issued
            # while a Gmail list is on screen does supersede it here -
            # unlike `current_email`/`current_thread`/`current_draft`
            # (Gmail's own fields, never touched by any `youtube.*`
            # branch), this one shared field is the one piece of Gmail
            # context Task 15 does not claim to preserve across a
            # YouTube search, by existing design.
            if tool_name == "youtube.search":
                state.current_search_results = list(data.get("results") or [])
            elif tool_name == "youtube.get_video" and data.get("results"):
                state.current_youtube_video = data["results"][0]
                if len(data["results"]) > 1:
                    state.current_search_results = list(data["results"])

            if tool_name == "youtube.video.update" and data.get("video"):
                state.current_youtube_video = data["video"]
            if tool_name == "youtube.video.delete":
                state.current_youtube_video = None

            if tool_name.startswith("youtube.playlist") and data.get("playlist"):
                state.current_playlist = data["playlist"]
            if tool_name == "youtube.playlist.delete":
                state.current_playlist = None
            if tool_name == "youtube.playlist.list":
                # Same generic "last list shown" slot as youtube.search /
                # gmail.search / youtube.channel.my_uploads, so "open the
                # second one" resolves against whichever list was shown last.
                state.current_search_results = list(data.get("playlists") or [])
            if tool_name == "youtube.playlist.get" and data.get("items"):
                state.current_search_results = list(data["items"])
            if tool_name == "youtube.channel.my_uploads":
                # Reuses the same generic "last list shown" slot as
                # youtube.search/gmail.search (see comment below) so "play
                # the second one" works right after "show my uploads" too.
                state.current_search_results = list(data.get("uploads") or [])

            if tool_name in ("gmail.get_message",) and data.get("message"):
                state.current_email = data["message"]
            elif tool_name in (
                "gmail.mark_read",
                "gmail.mark_unread",
                "gmail.archive",
                "gmail.star",
                "gmail.unstar",
                "gmail.add_label",
                "gmail.remove_label",
                "gmail.reply",
                "gmail.reply_all",
                "gmail.forward",
            ) and data.get("message"):
                state.current_email = data["message"]
            elif tool_name in ("gmail.search", "gmail.list_messages") and data.get("messages"):
                # Reuse current_search_results as the generic "last list
                # shown" slot so "open the second one" / "read the first
                # one" works for either YouTube or Gmail results without a
                # second list field. `gmail.list_messages` is what "show my
                # latest emails" (Task 60) resolves to, and its results need
                # to be resolvable by a follow-up just like a search does.
                state.current_search_results = list(data.get("messages") or [])

            if tool_name == "gmail.get_thread" and data.get("thread"):
                state.current_thread = data["thread"]

            if (tool_name.startswith("gmail.draft") or tool_name in ("ai.email.draft", "ai.email.suggest_reply")) and data.get("draft"):
                state.current_draft = data["draft"]
            if tool_name in ("gmail.draft.delete", "gmail.draft.send"):
                state.current_draft = None
