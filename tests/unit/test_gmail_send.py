"""Tests for Gmail *send* and its confirmation gate (Task 36: verify Gmail
send confirmation).

The requirement (see roadmap Phase 5/9, master spec section 39/10) is that
an AI-generated or user-composed email is never sent by surprise: the flow
must be *prepared email -> explicit confirmation -> send*, never
*request -> send*.

`gmail.send` (`SendMessageTool`) and `gmail.draft.send` (`SendDraftTool`)
had zero test coverage before this task - unlike `gmail.draft.create`/
`update`/`delete` (Task 31) and `gmail.reply`/`reply_all`/`forward` (Tasks
32-34), nothing exercised the two tools that actually put a message on the
wire, nor the specific confirmation-gate behavior for them. This file
closes that gap in two layers:

1. Tool-level: `SendMessageTool`/`SendDraftTool` validation, correct
   `GmailApiClient` calls, and `requires_confirmation = True`.
2. End-to-end through `PlannerOrchestrator`, using the project's existing
   scripted-LLM test harness (see `test_planner_orchestrator.py`) but with
   the *real* `SendMessageTool`, not a dummy stand-in - proving the
   generic confirmation gate actually holds a real, sensitive Gmail tool:
   the tool must not execute until a second turn with `confirm=True`, and
   must not execute at all if the user declines.
"""

from unittest.mock import MagicMock

import pytest

from app.agent.context import AgentRequestContext
from app.agent.orchestrator import LegacyRuleBasedOrchestrator
from app.agent.planner_orchestrator import PlannerOrchestrator
from app.agent.state import AgentStateStore
from app.core.errors import ValidationError
from app.llm.base import LLMResponse, LLMProvider, ProviderHealth, ToolCall
from app.tools.gmail.compose import SendDraftTool, SendMessageTool
from app.tools.registry import ToolRegistry


# -- SendMessageTool (gmail.send) --------------------------------------------


def test_send_message_requires_confirmation():
    assert SendMessageTool.requires_confirmation is True


def test_send_message_rejects_missing_recipients():
    tool = SendMessageTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"to": [], "subject": "Hi", "body_text": "Hello", "access_token": "tkn"})
    with pytest.raises(ValidationError):
        tool.execute({"to": ["not-an-email"], "subject": "Hi", "body_text": "Hello", "access_token": "tkn"})


def test_send_message_requires_access_token():
    tool = SendMessageTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"to": ["a@example.com"], "subject": "Hi", "body_text": "Hello"})


def test_send_message_calls_client_send_message_with_no_thread_headers():
    client = MagicMock()
    client.send_message.return_value = MagicMock(to_dict=lambda: {"id": "m1", "labelIds": ["SENT"]})
    tool = SendMessageTool(client=client)

    result = tool.execute(
        {"to": ["a@example.com"], "cc": ["b@example.com"], "subject": "Hi", "body_text": "Hello", "access_token": "tkn"}
    )

    assert result.success is True
    assert result.data == {"message": {"id": "m1", "labelIds": ["SENT"]}}
    assert client.send_message.call_args.kwargs == {
        "access_token": "tkn",
        "to": ["a@example.com"],
        "subject": "Hi",
        "body_text": "Hello",
        "cc": ["b@example.com"],
        "bcc": None,
    }
    # A direct send starts a new conversation - no threading headers.
    assert "thread_id" not in client.send_message.call_args.kwargs
    assert "in_reply_to" not in client.send_message.call_args.kwargs


# -- SendDraftTool (gmail.draft.send) ----------------------------------------


def test_send_draft_requires_confirmation():
    assert SendDraftTool.requires_confirmation is True


def test_send_draft_requires_draft_id():
    tool = SendDraftTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"draft_id": "", "access_token": "tkn"})


def test_send_draft_requires_access_token():
    tool = SendDraftTool(client=MagicMock())
    with pytest.raises(ValidationError):
        tool.execute({"draft_id": "d1"})


def test_send_draft_calls_client_send_draft():
    client = MagicMock()
    client.send_draft.return_value = MagicMock(to_dict=lambda: {"id": "m1", "labelIds": ["SENT"]})
    tool = SendDraftTool(client=client)

    result = tool.execute({"draft_id": "d1", "access_token": "tkn"})

    assert result.success is True
    client.send_draft.assert_called_once_with(access_token="tkn", draft_id="d1")


# -- End-to-end confirmation gate, with the real production tool ------------


class _ScriptedLLM(LLMProvider):
    name = "scripted"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, messages, *, tools=None, response_schema=None, timeout=20.0):
        self.calls.append(messages)
        return self._responses.pop(0)

    def health_check(self):
        return ProviderHealth(available=True, provider_name=self.name)


def _make_orchestrator(llm, registry):
    return PlannerOrchestrator(
        llm_provider=llm,
        tool_registry=registry,
        state_store=AgentStateStore(),
        legacy_fallback=LegacyRuleBasedOrchestrator(tool_registry=registry),
        max_tool_iterations=5,
    )


def _registry_with_real_send_tool(client):
    registry = ToolRegistry()
    registry.register(SendMessageTool(client=client))
    return registry


def test_gmail_send_is_held_for_confirmation_and_not_executed_immediately():
    client = MagicMock()
    registry = _registry_with_real_send_tool(client)
    scripted = _ScriptedLLM(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="gmail.send",
                        arguments={"to": ["jane@example.com"], "subject": "Hi", "body_text": "Hello Jane"},
                    )
                ],
            )
        ]
    )
    orchestrator = _make_orchestrator(scripted, registry)
    ctx = AgentRequestContext(session_id="s-send-1", access_token="tkn")

    result = orchestrator.handle("email jane saying hello", context=ctx)

    assert result.action_type == "confirmation_required"
    assert result.success is True
    assert {a["id"] for a in result.actions} == {"confirm", "cancel"}
    # The email must NOT have been sent yet.
    client.send_message.assert_not_called()

    state = orchestrator._state_store.get_or_create("s-send-1")
    assert state.pending_confirmation is not None
    assert state.pending_confirmation.tool_name == "gmail.send"


def test_gmail_send_only_fires_after_explicit_confirm_true():
    client = MagicMock()
    client.send_message.return_value = MagicMock(to_dict=lambda: {"id": "m1", "labelIds": ["SENT"]})
    registry = _registry_with_real_send_tool(client)
    scripted = _ScriptedLLM(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="gmail.send",
                        arguments={"to": ["jane@example.com"], "subject": "Hi", "body_text": "Hello Jane"},
                    )
                ],
            )
        ]
    )
    orchestrator = _make_orchestrator(scripted, registry)
    propose_ctx = AgentRequestContext(session_id="s-send-2", access_token="tkn")
    orchestrator.handle("email jane saying hello", context=propose_ctx)
    client.send_message.assert_not_called()

    confirm_ctx = AgentRequestContext(session_id="s-send-2", access_token="tkn", confirm=True)
    result = orchestrator.handle("", context=confirm_ctx)

    client.send_message.assert_called_once()
    assert client.send_message.call_args.kwargs["to"] == ["jane@example.com"]
    assert result.success is True

    state = orchestrator._state_store.get_or_create("s-send-2")
    assert state.pending_confirmation is None


def test_gmail_send_never_fires_if_user_declines():
    client = MagicMock()
    registry = _registry_with_real_send_tool(client)
    scripted = _ScriptedLLM(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="gmail.send",
                        arguments={"to": ["jane@example.com"], "subject": "Hi", "body_text": "Hello Jane"},
                    )
                ],
            )
        ]
    )
    orchestrator = _make_orchestrator(scripted, registry)
    propose_ctx = AgentRequestContext(session_id="s-send-3", access_token="tkn")
    orchestrator.handle("email jane saying hello", context=propose_ctx)

    decline_ctx = AgentRequestContext(session_id="s-send-3", access_token="tkn", confirm=False)
    result = orchestrator.handle("", context=decline_ctx)

    client.send_message.assert_not_called()
    assert result.success is True

    state = orchestrator._state_store.get_or_create("s-send-3")
    assert state.pending_confirmation is None
