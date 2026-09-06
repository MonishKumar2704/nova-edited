"""Gmail API request/response schemas (master spec Phase 6)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GmailListMessagesRequest(BaseModel):
    query: str | None = Field(None, max_length=500)
    label_ids: list[str] | None = None
    max_results: int = Field(25, ge=1, le=50)
    page_token: str | None = None


# --- Phase 7: message actions ------------------------------------------


class GmailLabelActionRequest(BaseModel):
    """Body for add-label/remove-label requests."""

    label_id: str = Field(..., min_length=1)


# --- Phase 7: composition ------------------------------------------------


class GmailComposeRequest(BaseModel):
    """Shared shape for draft create/update and direct send (master spec Phase 7/10)."""

    to: list[str] = Field(..., min_length=1)
    subject: str = Field("", max_length=500)
    body_text: str = Field("", max_length=100_000)
    cc: list[str] | None = None
    bcc: list[str] | None = None


# --- Phase 7: conversations ----------------------------------------------


class GmailReplyRequest(BaseModel):
    body_text: str = Field(..., min_length=1, max_length=100_000)
    extra_cc: str = Field("", description="Comma-separated extra Cc addresses for reply_all, appended to the original recipients.")


class GmailForwardRequest(BaseModel):
    to: list[str] = Field(..., min_length=1)
    body_text: str = Field("", max_length=100_000)
