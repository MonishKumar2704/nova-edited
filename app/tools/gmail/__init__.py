"""
Gmail tools: search, message/thread retrieval, labels (Phase 6); message
actions, composition, conversations, and attachments (Phase 7).

Phase 6 (Gmail API Foundation) added the read-only foundation:
`gmail.list_messages`, `gmail.search`, `gmail.get_message`,
`gmail.list_threads`, `gmail.get_thread`, and `gmail.list_labels`, all
backed by the official Gmail API v1 (`app.integrations.gmail_api`).

Phase 7 (Complete Gmail Operations) adds, as plain new `Tool` subclasses
registered the same way - zero changes to the orchestrator or the registry
itself (master spec section 7):
  - message actions (`app.tools.gmail.actions`): mark read/unread, archive,
    trash/untrash, star/unstar, add/remove label
  - composition (`app.tools.gmail.compose`): create/update/delete/list
    drafts, send a draft, compose-and-send directly
  - conversations (`app.tools.gmail.conversations`): reply, reply-all,
    forward
  - attachments (`app.tools.gmail.attachments`): download attachment bytes
"""

from __future__ import annotations

from app.integrations.gmail_api import GmailApiClient
from app.tools.gmail.actions import (
    AddLabelTool,
    ArchiveMessageTool,
    MarkReadTool,
    MarkUnreadTool,
    RemoveLabelTool,
    StarMessageTool,
    TrashMessageTool,
    UnstarMessageTool,
    UntrashMessageTool,
)
from app.tools.gmail.attachments import GetAttachmentTool
from app.tools.gmail.compose import (
    CreateDraftTool,
    DeleteDraftTool,
    ListDraftsTool,
    SendDraftTool,
    SendMessageTool,
    UpdateDraftTool,
)
from app.tools.gmail.conversations import ForwardTool, ReplyAllTool, ReplyTool
from app.tools.gmail.get_message import GetMessageTool
from app.tools.gmail.labels import ListLabelsTool
from app.tools.gmail.list_messages import ListMessagesTool, SearchMessagesTool
from app.tools.gmail.threads import GetThreadTool, ListThreadsTool
from app.tools.registry import ToolRegistry


def register_gmail_tools(registry: ToolRegistry, *, client: GmailApiClient) -> None:
    # Phase 6: read-only foundation
    registry.register(ListMessagesTool(client=client))
    registry.register(SearchMessagesTool(client=client))
    registry.register(GetMessageTool(client=client))
    registry.register(ListThreadsTool(client=client))
    registry.register(GetThreadTool(client=client))
    registry.register(ListLabelsTool(client=client))

    # Phase 7: message actions
    registry.register(MarkReadTool(client=client))
    registry.register(MarkUnreadTool(client=client))
    registry.register(ArchiveMessageTool(client=client))
    registry.register(TrashMessageTool(client=client))
    registry.register(UntrashMessageTool(client=client))
    registry.register(StarMessageTool(client=client))
    registry.register(UnstarMessageTool(client=client))
    registry.register(AddLabelTool(client=client))
    registry.register(RemoveLabelTool(client=client))

    # Phase 7: composition
    registry.register(CreateDraftTool(client=client))
    registry.register(UpdateDraftTool(client=client))
    registry.register(DeleteDraftTool(client=client))
    registry.register(ListDraftsTool(client=client))
    registry.register(SendDraftTool(client=client))
    registry.register(SendMessageTool(client=client))

    # Phase 7: conversations
    registry.register(ReplyTool(client=client))
    registry.register(ReplyAllTool(client=client))
    registry.register(ForwardTool(client=client))

    # Phase 7: attachments
    registry.register(GetAttachmentTool(client=client))
