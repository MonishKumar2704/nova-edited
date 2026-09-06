"""
Lightweight sanity checks for `app.services.dynamic_ui` (Phase 4).

Full coverage of every card/action type is Phase 17's job (master spec
section 3) - this just confirms the envelope shape the frontend
(`static/js/dynamic-actions.js`) depends on doesn't silently drift.
"""

from __future__ import annotations

from app.services.dynamic_ui import action, build_video_card, build_video_cards


def test_action_shape():
    a = action("play", "Play")
    assert a == {"id": "play", "label": "Play", "requires_confirmation": False}

    confirmed = action("add_to_playlist", "Add to Playlist", requires_confirmation=True)
    assert confirmed["requires_confirmation"] is True


def test_build_video_card_envelope():
    video = {"video_id": "abc123", "title": "Test Video", "channel_title": "Test Channel"}
    card = build_video_card(video)

    assert card["type"] == "youtube_video"
    assert card["data"] == video
    action_ids = [a["id"] for a in card["actions"]]
    assert action_ids == ["play", "open", "queue", "add_to_playlist", "share"]


def test_build_video_card_actions_are_not_shared_mutable_state():
    card_a = build_video_card({"video_id": "a"})
    card_b = build_video_card({"video_id": "b"})

    card_a["actions"].append({"id": "custom", "label": "Custom", "requires_confirmation": False})

    assert [a["id"] for a in card_b["actions"]] == ["play", "open", "queue", "add_to_playlist", "share"]


def test_build_video_cards_list():
    videos = [{"video_id": "a"}, {"video_id": "b"}]
    cards = build_video_cards(videos)

    assert len(cards) == 2
    assert all(c["type"] == "youtube_video" for c in cards)
    assert [c["data"]["video_id"] for c in cards] == ["a", "b"]
