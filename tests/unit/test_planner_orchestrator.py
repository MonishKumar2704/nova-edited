"""
Lightweight sanity tests for `PlannerOrchestrator` (Phase 9).

Per the project's testing policy, this is NOT the comprehensive test
suite (that's Phase 17) - just enough to catch an obviously broken
plan/act loop, confirmation gate, or fallback path using a fake
`LLMProvider` (no network/real Gemini/Ollama call involved).
"""

from __future__ import annotations

from app.agent.context import AgentRequestContext
from app.agent.orchestrator import LegacyRuleBasedOrchestrator
from app.agent.planner_orchestrator import PlannerOrchestrator
from app.agent.state import AgentState, AgentStateStore
from app.core.errors import LLMError, NetworkError, TimeoutErrorNova
from app.llm.base import LLMMessage, LLMProvider, LLMResponse, ProviderHealth, ToolCall
from app.tools.base import Tool, ToolResult
from app.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "dummy.echo"
    description = "Echoes back its input."
    input_schema = {"text": {"type": "string", "required": True}}
    output_schema = {"text": {"type": "string"}}
    requires_confirmation = False

    def execute(self, arguments):
        return ToolResult(success=True, data={"text": arguments.get("text", "")})


class DangerousTool(Tool):
    name = "dummy.delete_everything"
    description = "A destructive tool that must be confirmed first."
    input_schema = {"target": {"type": "string", "required": True}}
    output_schema = {"deleted": {"type": "boolean"}}
    requires_confirmation = True

    def execute(self, arguments):
        return ToolResult(success=True, data={"deleted": True, "target": arguments.get("target")})


class ScriptedLLMProvider(LLMProvider):
    """Returns pre-scripted `LLMResponse`s in order, one per `generate()` call."""

    name = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[LLMMessage]] = []

    def generate(self, messages, *, tools=None, response_schema=None, timeout=20.0) -> LLMResponse:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("ScriptedLLMProvider ran out of scripted responses")
        return self._responses.pop(0)

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(available=True, provider_name=self.name)


class UnavailableLLMProvider(LLMProvider):
    name = "unavailable"

    def generate(self, messages, *, tools=None, response_schema=None, timeout=20.0) -> LLMResponse:
        raise LLMError("No LLM provider is configured.")

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(available=False, provider_name=self.name, detail="not configured")


class OfflineOllamaProvider(LLMProvider):
    """Simulates a configured Ollama provider whose server isn't running.

    Unlike `UnavailableLLMProvider` (raises `LLMError`, the "no provider
    configured" case), this raises the classified errors the real
    `OllamaProvider` raises when it can't reach `localhost:11434` at all,
    or when the request times out (Task 58).
    """

    name = "ollama-offline"

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or NetworkError("Could not reach Ollama server at http://localhost:11434: refused.")

    def generate(self, messages, *, tools=None, response_schema=None, timeout=20.0) -> LLMResponse:
        raise self._exc

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(available=False, provider_name=self.name, detail="Ollama server unreachable")


def _make_orchestrator(llm, *, registry: ToolRegistry | None = None, max_iterations: int = 5) -> PlannerOrchestrator:
    registry = registry or ToolRegistry()
    return PlannerOrchestrator(
        llm_provider=llm,
        tool_registry=registry,
        state_store=AgentStateStore(),
        legacy_fallback=LegacyRuleBasedOrchestrator(tool_registry=registry),
        max_tool_iterations=max_iterations,
    )


def test_build_action_result_surfaces_channel_info():
    """`youtube.list_channels` must not fall through to the generic
    "Done." message - the whole point of the tool is to answer "what's
    my channel's info" (Task 42)."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.list_channels",
        ToolResult(
            success=True,
            data={
                "channel": {
                    "channel_id": "ch1",
                    "title": "My Channel",
                    "subscriber_count": 1234,
                    "video_count": 56,
                }
            },
        ),
    )

    assert result.success is True
    assert result.action_type == "youtube_channel"
    assert "My Channel" in result.message
    assert "1,234" in result.message
    assert result.data["channel"]["channel_id"] == "ch1"


def test_build_action_result_channel_without_subscriber_count():
    """Subscriber count can be hidden by the account's privacy settings
    (`None` from `ChannelSummary`) - must still produce a readable
    message, not a crash or a literal "None"."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.list_channels",
        ToolResult(success=True, data={"channel": {"channel_id": "ch1", "title": "My Channel"}}),
    )

    assert result.success is True
    assert result.message == "Channel: My Channel."


def test_build_action_result_surfaces_uploads():
    """`youtube.channel.my_uploads` must not fall through to the generic
    "Done." message either (Task 43 - same gap class as Task 42's
    youtube.list_channels)."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    uploads = [
        {"playlist_item_id": "pi1", "video_id": "vid1", "title": "First upload"},
        {"playlist_item_id": "pi2", "video_id": "vid2", "title": "Second upload"},
    ]
    result = orchestrator._build_action_result(
        "youtube.channel.my_uploads",
        ToolResult(success=True, data={"uploads": uploads, "next_page_token": None}),
    )

    assert result.success is True
    assert result.action_type == "youtube_uploads"
    assert result.message == "Found 2 of your uploaded videos."
    assert result.data["card"]["data"]["video_id"] == "vid1"


def test_build_action_result_uploads_singular_wording():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.channel.my_uploads",
        ToolResult(success=True, data={"uploads": [{"playlist_item_id": "pi1", "video_id": "vid1"}]}),
    )

    assert result.message == "Found 1 of your uploaded video."


def test_build_action_result_no_uploads():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.channel.my_uploads", ToolResult(success=True, data={"uploads": []})
    )

    assert result.success is True
    assert result.action_type == "youtube_uploads"
    assert result.message == "You don't have any uploaded videos."
    assert "card" not in result.data


def test_update_state_stores_uploads_as_current_search_results():
    """So a follow-up like "play the second one" can resolve against the
    uploads list the same way it already does for youtube.search /
    gmail.search results."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    state = AgentState()
    uploads = [{"video_id": "vid1"}, {"video_id": "vid2"}]

    orchestrator._update_state_from_tool_result(
        state, "youtube.channel.my_uploads", ToolResult(success=True, data={"uploads": uploads})
    )

    assert state.current_search_results == uploads


def test_build_action_result_surfaces_playlist_list():
    """`youtube.playlist.list` must not fall through to the generic
    "Done." message either (Task 44 - same gap class as Tasks 42/43):
    it has no top-level "playlist" key, only "playlists", so the
    existing create/update/get branch never matched it."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    playlists = [
        {"playlist_id": "PL1", "title": "Watch Later"},
        {"playlist_id": "PL2", "title": "Favorites"},
    ]
    result = orchestrator._build_action_result(
        "youtube.playlist.list",
        ToolResult(success=True, data={"playlists": playlists, "next_page_token": None}),
    )

    assert result.success is True
    assert result.action_type == "youtube_playlists"
    assert result.message == "Found 2 playlists."
    assert result.data["cards"][0]["data"]["playlist_id"] == "PL1"


def test_build_action_result_playlist_list_singular_wording():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.playlist.list",
        ToolResult(success=True, data={"playlists": [{"playlist_id": "PL1", "title": "Watch Later"}]}),
    )

    assert result.message == "Found 1 playlist."


def test_build_action_result_no_playlists():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.playlist.list", ToolResult(success=True, data={"playlists": []})
    )

    assert result.success is True
    assert result.action_type == "youtube_playlists"
    assert result.message == "You don't have any playlists."
    assert "cards" not in result.data


def test_build_action_result_playlist_get_includes_item_count():
    """`youtube.playlist.get` shares the generic playlist branch's
    "ready." message today, but it also returns `items` (the videos
    inside the playlist) that were previously dropped entirely - no
    card, no count, no way to say "play the second one" afterwards."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    items = [
        {"playlist_item_id": "pi1", "video_id": "vid1"},
        {"playlist_item_id": "pi2", "video_id": "vid2"},
    ]
    result = orchestrator._build_action_result(
        "youtube.playlist.get",
        ToolResult(
            success=True,
            data={"playlist": {"playlist_id": "PL1", "title": "Watch Later"}, "items": items, "next_page_token": None},
        ),
    )

    assert result.success is True
    assert result.action_type == "youtube_playlist"
    assert result.message == "Playlist 'Watch Later' (2 videos)."
    assert result.data["items"] == items
    assert result.data["cards"][0]["data"]["video_id"] == "vid1"


def test_build_action_result_playlist_get_empty_playlist():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.playlist.get",
        ToolResult(success=True, data={"playlist": {"playlist_id": "PL1", "title": "Empty"}, "items": []}),
    )

    assert result.message == "Playlist 'Empty'."


def test_build_action_result_surfaces_created_playlist():
    """`youtube.playlist.create` shares the generic write-branch (it has a
    top-level "playlist" key), so it should produce a ready card/message
    instead of falling through to "Done."."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.playlist.create",
        ToolResult(success=True, data={"playlist": {"playlist_id": "PL1", "title": "My Mix"}}),
    )

    assert result.success is True
    assert result.action_type == "youtube_playlist"
    assert result.message == "Playlist 'My Mix' ready."
    assert result.data["card"]["data"]["playlist_id"] == "PL1"


def test_update_state_stores_created_playlist_as_current_playlist():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    state = AgentState()

    orchestrator._update_state_from_tool_result(
        state,
        "youtube.playlist.create",
        ToolResult(success=True, data={"playlist": {"playlist_id": "PL1", "title": "My Mix"}}),
    )

    assert state.current_playlist == {"playlist_id": "PL1", "title": "My Mix"}


def test_build_action_result_surfaces_updated_playlist():
    """`youtube.playlist.update` shares the same generic write-branch as
    `.create` (both return a top-level "playlist" key)."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.playlist.update",
        ToolResult(success=True, data={"playlist": {"playlist_id": "PL1", "title": "Renamed"}}),
    )

    assert result.success is True
    assert result.action_type == "youtube_playlist"
    assert result.message == "Playlist 'Renamed' ready."
    assert result.data["card"]["data"]["playlist_id"] == "PL1"


def test_update_state_stores_updated_playlist_as_current_playlist():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    state = AgentState()

    orchestrator._update_state_from_tool_result(
        state,
        "youtube.playlist.update",
        ToolResult(success=True, data={"playlist": {"playlist_id": "PL1", "title": "Renamed"}}),
    )

    assert state.current_playlist == {"playlist_id": "PL1", "title": "Renamed"}


def test_build_action_result_surfaces_added_playlist_item():
    """`youtube.playlist.add_video` returns an "item" key, not "playlist",
    so it never matched the shared playlist-write branch and fell through
    to "Done." - same gap class as the delete-playlist fix in Task 47."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.playlist.add_video",
        ToolResult(success=True, data={"item": {"playlist_item_id": "pi1", "video_id": "vid1", "title": "First"}}),
    )

    assert result.success is True
    assert result.action_type == "youtube_playlist_item_added"
    assert result.message == "Added 'First' to the playlist."


def test_build_action_result_surfaces_removed_playlist_item():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.playlist.remove_video",
        ToolResult(success=True, data={"removed": True, "playlist_item_id": "pi1"}),
    )

    assert result.success is True
    assert result.action_type == "youtube_playlist_item_removed"
    assert result.message == "Removed from the playlist."


def test_build_action_result_surfaces_reordered_playlist_item():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.playlist.reorder_video",
        ToolResult(success=True, data={"item": {"playlist_item_id": "pi1", "video_id": "vid1", "title": "First", "position": 2}}),
    )

    assert result.success is True
    assert result.action_type == "youtube_playlist_item_reordered"
    assert result.message == "Moved 'First' to position 3."


def test_build_action_result_surfaces_deleted_playlist():
    """`youtube.playlist.delete` has no top-level "playlist" key (only
    "deleted"/"playlist_id"), so it fell all the way through to the
    generic "Done." / action_type="tool_result" fallback - same gap class
    as Tasks 42/43/44. Fixed with a dedicated branch."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.playlist.delete",
        ToolResult(success=True, data={"deleted": True, "playlist_id": "PL1"}),
    )

    assert result.success is True
    assert result.action_type == "youtube_playlist_deleted"
    assert result.message == "Playlist deleted."


def test_update_state_clears_current_playlist_on_delete():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    state = AgentState()
    state.current_playlist = {"playlist_id": "PL1"}

    orchestrator._update_state_from_tool_result(
        state,
        "youtube.playlist.delete",
        ToolResult(success=True, data={"deleted": True, "playlist_id": "PL1"}),
    )

    assert state.current_playlist is None


def test_build_action_result_surfaces_updated_video():
    """`youtube.video.update` returns a top-level "video" key, which never
    matched any existing branch (the playlist branches all key off
    "playlist"), so it fell through to the generic "Done." fallback -
    same gap class as Tasks 42-48."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.video.update",
        ToolResult(success=True, data={"video": {"video_id": "vid1", "title": "New Title"}}),
    )

    assert result.success is True
    assert result.action_type == "youtube_video_updated"
    assert result.message == "Updated 'New Title'."
    assert result.data["card"]["data"]["video_id"] == "vid1"


def test_build_action_result_surfaces_deleted_video():
    """`youtube.video.delete` has no top-level "video" key (only
    "deleted"/"video_id"), so it fell through to the generic "Done." /
    action_type="tool_result" fallback - same gap class as the
    delete-playlist fix in Task 47."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.video.delete",
        ToolResult(success=True, data={"deleted": True, "video_id": "vid1"}),
    )

    assert result.success is True
    assert result.action_type == "youtube_video_deleted"
    assert result.message == "Video deleted."


def test_update_state_stores_updated_video_as_current_video():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    state = AgentState()

    orchestrator._update_state_from_tool_result(
        state,
        "youtube.video.update",
        ToolResult(success=True, data={"video": {"video_id": "vid1", "title": "New Title"}}),
    )

    assert state.current_youtube_video == {"video_id": "vid1", "title": "New Title"}


def test_update_state_clears_current_video_on_delete():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    state = AgentState()
    state.current_youtube_video = {"video_id": "vid1"}

    orchestrator._update_state_from_tool_result(
        state,
        "youtube.video.delete",
        ToolResult(success=True, data={"deleted": True, "video_id": "vid1"}),
    )

    assert state.current_youtube_video is None


def test_gmail_activity_preserves_current_youtube_video_state():
    """Task 14: opening/using Gmail must not disturb whatever YouTube
    state a session already has - `current_youtube_video` lives in its
    own `AgentState` field (see app.agent.state), separate from every
    Gmail field, specifically so "play a video, then open Gmail, then
    come back to the video" doesn't lose the video Nova had playing.
    """
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    state = AgentState()
    state.current_youtube_video = {"video_id": "vid1", "title": "Now Playing"}

    # A representative spread of Gmail tool results a session might see
    # while "Gmail is open" - none of them should touch the video field.
    orchestrator._update_state_from_tool_result(
        state,
        "gmail.get_message",
        ToolResult(success=True, data={"message": {"message_id": "m1", "subject": "Hi"}}),
    )
    assert state.current_youtube_video == {"video_id": "vid1", "title": "Now Playing"}

    orchestrator._update_state_from_tool_result(
        state,
        "gmail.list_messages",
        ToolResult(success=True, data={"messages": [{"message_id": "m1"}, {"message_id": "m2"}]}),
    )
    assert state.current_youtube_video == {"video_id": "vid1", "title": "Now Playing"}

    orchestrator._update_state_from_tool_result(
        state,
        "gmail.draft.create",
        ToolResult(success=True, data={"draft": {"draft_id": "d1", "message": {"subject": "Draft"}}}),
    )
    assert state.current_youtube_video == {"video_id": "vid1", "title": "Now Playing"}

    orchestrator._update_state_from_tool_result(
        state,
        "gmail.get_thread",
        ToolResult(success=True, data={"thread": {"thread_id": "t1", "messages": []}}),
    )
    assert state.current_youtube_video == {"video_id": "vid1", "title": "Now Playing"}


def test_youtube_activity_preserves_current_gmail_state():
    """Task 15: the converse of Task 14 above - switching to YouTube (and
    using it) must not disturb whatever Gmail context a session already
    has. `current_email`/`current_thread`/`current_draft` are their own
    `AgentState` fields (see app.agent.state), untouched by any
    `youtube.*` branch of `_update_state_from_tool_result`, specifically
    so "open an email, then open YouTube and play something, then come
    back to Gmail" doesn't lose the previously opened email/thread/draft.
    """
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    state = AgentState()
    state.current_email = {"message_id": "m1", "subject": "Previously opened"}
    state.current_thread = {"thread_id": "t1", "messages": [{"message_id": "m1"}]}
    state.current_draft = {"draft_id": "d1", "message": {"subject": "In progress"}}

    # A representative spread of YouTube tool results a session might see
    # while "YouTube is open" - none of them should touch the Gmail fields.
    orchestrator._update_state_from_tool_result(
        state,
        "youtube.get_video",
        ToolResult(success=True, data={"results": [{"video_id": "vid1", "title": "Now Playing"}]}),
    )
    assert state.current_email == {"message_id": "m1", "subject": "Previously opened"}
    assert state.current_thread == {"thread_id": "t1", "messages": [{"message_id": "m1"}]}
    assert state.current_draft == {"draft_id": "d1", "message": {"subject": "In progress"}}

    orchestrator._update_state_from_tool_result(
        state,
        "youtube.search",
        ToolResult(success=True, data={"results": [{"video_id": "vid2"}, {"video_id": "vid3"}]}),
    )
    assert state.current_email == {"message_id": "m1", "subject": "Previously opened"}
    assert state.current_thread == {"thread_id": "t1", "messages": [{"message_id": "m1"}]}
    assert state.current_draft == {"draft_id": "d1", "message": {"subject": "In progress"}}

    orchestrator._update_state_from_tool_result(
        state,
        "youtube.video.update",
        ToolResult(success=True, data={"video": {"video_id": "vid1", "title": "Renamed"}}),
    )
    assert state.current_email == {"message_id": "m1", "subject": "Previously opened"}
    assert state.current_thread == {"thread_id": "t1", "messages": [{"message_id": "m1"}]}
    assert state.current_draft == {"draft_id": "d1", "message": {"subject": "In progress"}}

    orchestrator._update_state_from_tool_result(
        state,
        "youtube.video.delete",
        ToolResult(success=True, data={"deleted": True, "video_id": "vid1"}),
    )
    assert state.current_email == {"message_id": "m1", "subject": "Previously opened"}
    assert state.current_thread == {"thread_id": "t1", "messages": [{"message_id": "m1"}]}
    assert state.current_draft == {"draft_id": "d1", "message": {"subject": "In progress"}}


def test_build_action_result_surfaces_video_rating_like():
    """`youtube.video.rate` returns a top-level "rating" key, which never
    matched any existing branch, so it fell through to the generic
    "Done." fallback - same gap class as Tasks 42-49."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.video.rate",
        ToolResult(success=True, data={"video_id": "vid1", "rating": "like"}),
    )

    assert result.success is True
    assert result.action_type == "youtube_video_rated"
    assert result.message == "Liked the video."


def test_build_action_result_surfaces_video_rating_cleared():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.video.rate",
        ToolResult(success=True, data={"video_id": "vid1", "rating": "none"}),
    )

    assert result.action_type == "youtube_video_rated"
    assert result.message == "Cleared your rating on the video."


def test_build_action_result_surfaces_get_video_rating():
    """`youtube.video.get_rating` also returns a top-level "rating" key -
    same gap class, distinguished from `youtube.video.rate` by tool name."""
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.video.get_rating",
        ToolResult(success=True, data={"video_id": "vid1", "rating": "dislike"}),
    )

    assert result.success is True
    assert result.action_type == "youtube_video_rating"
    assert result.message == "Your current rating on this video: dislike."


def test_build_action_result_surfaces_get_video_rating_none():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "youtube.video.get_rating",
        ToolResult(success=True, data={"video_id": "vid1", "rating": None}),
    )

    assert result.message == "Your current rating on this video: none."


def test_build_action_result_surfaces_generated_email():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "ai.email.generate",
        ToolResult(success=True, data={"subject": "Extension request", "body": "Dear Professor, ..."}),
    )

    assert result.success is True
    assert result.action_type == "ai_email_generated"
    assert "Extension request" in result.message
    assert "Dear Professor" in result.message
    assert result.data == {"subject": "Extension request", "body": "Dear Professor, ..."}


def test_build_action_result_surfaces_ai_email_draft():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "ai.email.draft",
        ToolResult(
            success=True,
            data={
                "draft": {"draft_id": "d1", "message": {"subject": "Extension request"}},
                "subject": "Extension request",
                "body": "Dear Professor, ...",
            },
        ),
    )

    assert result.success is True
    assert result.action_type == "ai_email_drafted"
    assert "Extension request" in result.message
    assert result.data["draft"]["draft_id"] == "d1"


def test_update_state_stores_ai_generated_draft_as_current_draft():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    state = AgentState()

    orchestrator._update_state_from_tool_result(
        state,
        "ai.email.draft",
        ToolResult(success=True, data={"draft": {"draft_id": "d1"}, "subject": "S", "body": "B"}),
    )

    assert state.current_draft == {"draft_id": "d1"}


def test_build_action_result_surfaces_rewritten_email():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    result = orchestrator._build_action_result(
        "ai.email.rewrite",
        ToolResult(success=True, data={"text": "Dear team, ...", "style": "professional"}),
    )

    assert result.success is True
    assert result.action_type == "ai_email_rewritten"
    assert "Dear team, ..." in result.message


def test_update_state_stores_playlist_list_as_current_search_results():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    state = AgentState()
    playlists = [{"playlist_id": "PL1"}, {"playlist_id": "PL2"}]

    orchestrator._update_state_from_tool_result(
        state, "youtube.playlist.list", ToolResult(success=True, data={"playlists": playlists})
    )

    assert state.current_search_results == playlists


def test_update_state_stores_playlist_get_items_as_current_search_results():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    state = AgentState()
    items = [{"video_id": "vid1"}, {"video_id": "vid2"}]

    orchestrator._update_state_from_tool_result(
        state,
        "youtube.playlist.get",
        ToolResult(success=True, data={"playlist": {"playlist_id": "PL1"}, "items": items}),
    )

    assert state.current_search_results == items
    assert state.current_playlist == {"playlist_id": "PL1"}


def test_falls_back_to_legacy_router_when_llm_unavailable():
    orchestrator = _make_orchestrator(UnavailableLLMProvider())
    ctx = AgentRequestContext(session_id="s1")

    result = orchestrator.handle("email jane at gmail dot com type hi", context=ctx)

    assert result.success is True
    assert result.action_type == "gmail_compose"


def test_falls_back_to_legacy_router_when_ollama_unreachable():
    """Ollama being down (`NetworkError`, not `LLMError`) must degrade the
    same way "no provider configured" does - Gmail/YouTube keyword
    commands keep working (Task 58)."""
    orchestrator = _make_orchestrator(OfflineOllamaProvider())
    ctx = AgentRequestContext(session_id="s-offline-1")

    result = orchestrator.handle("email jane at gmail dot com type hi", context=ctx)

    assert result.success is True
    assert result.action_type == "gmail_compose"


def test_ollama_unreachable_gives_clear_start_ollama_message_for_unmatched_commands():
    """A command the legacy router also can't route (most likely because
    it needed the AI itself) must not surface as a raw crash/unknown
    error - it should plainly say Ollama needs to be started, per Task 58."""
    orchestrator = _make_orchestrator(OfflineOllamaProvider())
    ctx = AgentRequestContext(session_id="s-offline-2")

    result = orchestrator.handle("write a short poem about the sea", context=ctx)

    assert result.success is False
    assert "ollama" in result.message.lower()


def test_ollama_timeout_also_falls_back_without_crashing():
    orchestrator = _make_orchestrator(OfflineOllamaProvider(TimeoutErrorNova("Ollama request timed out after 20.0s.")))
    ctx = AgentRequestContext(session_id="s-offline-3")

    result = orchestrator.handle("email jane at gmail dot com type hi", context=ctx)

    assert result.success is True
    assert result.action_type == "gmail_compose"


def test_plan_loop_executes_tool_then_returns_final_text():
    registry = ToolRegistry()
    registry.register(EchoTool())

    scripted = ScriptedLLMProvider(
        [
            LLMResponse(text="", tool_calls=[ToolCall(id="1", name="dummy.echo", arguments={"text": "hello"})]),
            LLMResponse(text="I echoed 'hello' back.", tool_calls=[]),
        ]
    )
    orchestrator = _make_orchestrator(scripted, registry=registry)
    ctx = AgentRequestContext(session_id="s2")

    result = orchestrator.handle("echo hello", context=ctx)

    assert result.success is True
    assert result.action_type == "agent_reply"
    assert "echoed" in result.message
    assert len(scripted.calls) == 2  # one round-trip per tool call + one final


def test_requires_confirmation_tool_is_held_for_explicit_consent():
    registry = ToolRegistry()
    registry.register(DangerousTool())

    scripted = ScriptedLLMProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="1", name="dummy.delete_everything", arguments={"target": "everything"})],
            )
        ]
    )
    orchestrator = _make_orchestrator(scripted, registry=registry)
    ctx = AgentRequestContext(session_id="s3")

    result = orchestrator.handle("delete everything", context=ctx)

    assert result.action_type == "confirmation_required"
    assert result.success is True
    action_ids = {a["id"] for a in result.actions}
    assert action_ids == {"confirm", "cancel"}

    # The tool must NOT have run yet.
    state = orchestrator._state_store.get_or_create("s3")
    assert state.pending_confirmation is not None
    assert state.pending_confirmation.tool_name == "dummy.delete_everything"


def test_confirming_a_pending_action_executes_it():
    registry = ToolRegistry()
    registry.register(DangerousTool())

    scripted = ScriptedLLMProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="1", name="dummy.delete_everything", arguments={"target": "everything"})],
            )
        ]
    )
    orchestrator = _make_orchestrator(scripted, registry=registry)
    ctx = AgentRequestContext(session_id="s4")

    orchestrator.handle("delete everything", context=ctx)

    confirm_ctx = AgentRequestContext(session_id="s4", confirm=True)
    result = orchestrator.handle("", context=confirm_ctx)

    assert result.success is True
    state = orchestrator._state_store.get_or_create("s4")
    assert state.pending_confirmation is None


def test_declining_a_pending_action_clears_it_without_executing():
    registry = ToolRegistry()

    class TrackingTool(DangerousTool):
        executed = False

        def execute(self, arguments):
            TrackingTool.executed = True
            return super().execute(arguments)

    registry.register(TrackingTool())

    scripted = ScriptedLLMProvider(
        [LLMResponse(text="", tool_calls=[ToolCall(id="1", name="dummy.delete_everything", arguments={"target": "x"})])]
    )
    orchestrator = _make_orchestrator(scripted, registry=registry)
    ctx = AgentRequestContext(session_id="s5")
    orchestrator.handle("delete x", context=ctx)

    cancel_ctx = AgentRequestContext(session_id="s5", confirm=False)
    result = orchestrator.handle("", context=cancel_ctx)

    assert result.success is True
    assert TrackingTool.executed is False
    state = orchestrator._state_store.get_or_create("s5")
    assert state.pending_confirmation is None


def test_iteration_budget_is_bounded():
    registry = ToolRegistry()
    registry.register(EchoTool())

    # Always requests another tool call - the loop must not spin forever.
    scripted = ScriptedLLMProvider(
        [
            LLMResponse(text="", tool_calls=[ToolCall(id=str(i), name="dummy.echo", arguments={"text": "x"})])
            for i in range(10)
        ]
    )
    orchestrator = _make_orchestrator(scripted, registry=registry, max_iterations=3)
    ctx = AgentRequestContext(session_id="s6")

    result = orchestrator.handle("loop forever", context=ctx)

    assert result.success is False
    assert result.action_type == "agent_incomplete"
    assert len(scripted.calls) == 3


# -- Task 59: AI-generated emails must never auto-send -------------------------------------------------


class FakeAiDraftTool(Tool):
    """Stand-in for `ai.email.draft`: generates + saves a draft, never sends."""

    name = "ai.email.draft"
    description = "Generate an email and save it as a draft."
    input_schema = {"instruction": {"type": "string", "required": True}, "to": {"type": "array", "required": True}}
    output_schema = {"draft": {"type": "object"}}
    requires_confirmation = False

    def execute(self, arguments):
        draft = {
            "draft_id": "d1",
            "message": {
                "to": "prof@uni.edu",
                "subject": "Extension request",
                "body_text": "Dear Professor, requesting a short extension. Thanks.",
            },
        }
        return ToolResult(
            success=True,
            data={"draft": draft, "subject": "Extension request", "body": draft["message"]["body_text"]},
        )


class FakeDraftSendTool(Tool):
    """Stand-in for `gmail.draft.send`: the one tool that actually delivers an AI-drafted email."""

    name = "gmail.draft.send"
    description = "Send an existing Gmail draft as-is."
    input_schema = {"draft_id": {"type": "string", "required": True}}
    output_schema = {"message": {"type": "object"}}
    requires_confirmation = True

    def __init__(self):
        self.sent = False

    def execute(self, arguments):
        self.sent = True
        return ToolResult(success=True, data={"message": {"id": "m1"}})


def test_ai_generated_email_is_never_sent_without_confirmation():
    """The full Task 59 pipeline: request -> AI draft -> preview -> hold for
    confirmation. The send tool must not execute until the user confirms."""
    registry = ToolRegistry()
    registry.register(FakeAiDraftTool())
    send_tool = FakeDraftSendTool()
    registry.register(send_tool)

    scripted = ScriptedLLMProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="ai.email.draft",
                        arguments={"instruction": "ask for extension", "to": ["prof@uni.edu"]},
                    )
                ],
            ),
            LLMResponse(text="", tool_calls=[ToolCall(id="2", name="gmail.draft.send", arguments={"draft_id": "d1"})]),
        ]
    )
    orchestrator = _make_orchestrator(scripted, registry=registry)
    ctx = AgentRequestContext(session_id="s-ai-send-1", access_token="tok")

    result = orchestrator.handle("write an email to my professor asking for an extension", context=ctx)

    assert result.action_type == "confirmation_required"
    assert send_tool.sent is False


def test_ai_generated_email_confirmation_previews_subject_and_body():
    """The confirmation prompt must show what's actually about to be sent -
    not a content-free \"Send the saved draft.\" - so the user has a real
    preview to approve or reject (Task 59)."""
    registry = ToolRegistry()
    registry.register(FakeAiDraftTool())
    registry.register(FakeDraftSendTool())

    scripted = ScriptedLLMProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="ai.email.draft",
                        arguments={"instruction": "ask for extension", "to": ["prof@uni.edu"]},
                    )
                ],
            ),
            LLMResponse(text="", tool_calls=[ToolCall(id="2", name="gmail.draft.send", arguments={"draft_id": "d1"})]),
        ]
    )
    orchestrator = _make_orchestrator(scripted, registry=registry)
    ctx = AgentRequestContext(session_id="s-ai-send-2", access_token="tok")

    result = orchestrator.handle("write an email to my professor asking for an extension", context=ctx)

    assert "Extension request" in result.message
    assert "Dear Professor" in result.message
    assert "prof@uni.edu" in result.message


def test_ai_generated_email_sends_only_after_explicit_confirmation():
    registry = ToolRegistry()
    registry.register(FakeAiDraftTool())
    send_tool = FakeDraftSendTool()
    registry.register(send_tool)

    scripted = ScriptedLLMProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="ai.email.draft",
                        arguments={"instruction": "ask for extension", "to": ["prof@uni.edu"]},
                    )
                ],
            ),
            LLMResponse(text="", tool_calls=[ToolCall(id="2", name="gmail.draft.send", arguments={"draft_id": "d1"})]),
        ]
    )
    orchestrator = _make_orchestrator(scripted, registry=registry)
    session_id = "s-ai-send-3"
    orchestrator.handle(
        "write an email to my professor asking for an extension",
        context=AgentRequestContext(session_id=session_id, access_token="tok"),
    )
    assert send_tool.sent is False

    result = orchestrator.handle(
        "", context=AgentRequestContext(session_id=session_id, access_token="tok", confirm=True)
    )

    assert send_tool.sent is True
    assert result.success is True


def test_draft_send_confirmation_falls_back_to_generic_message_when_draft_untracked():
    """If `AgentState.current_draft` doesn't match the draft being sent (or
    isn't tracked at all), fall back to the safe generic description
    instead of ever showing the wrong content."""
    registry = ToolRegistry()
    send_tool = FakeDraftSendTool()
    registry.register(send_tool)

    scripted = ScriptedLLMProvider(
        [LLMResponse(text="", tool_calls=[ToolCall(id="1", name="gmail.draft.send", arguments={"draft_id": "unknown"})])]
    )
    orchestrator = _make_orchestrator(scripted, registry=registry)
    ctx = AgentRequestContext(session_id="s-ai-send-4", access_token="tok")

    result = orchestrator.handle("send that draft", context=ctx)

    assert result.action_type == "confirmation_required"
    assert result.message == "Send the saved draft. Shall I go ahead?"
    assert send_tool.sent is False
