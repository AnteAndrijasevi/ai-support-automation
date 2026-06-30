import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models import Category, ConfidenceFlag, Sentiment, Urgency


class TicketCreate(BaseModel):
    subject: str = Field(min_length=1, max_length=255)
    body: str = Field(min_length=1, max_length=20_000)


class TicketRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    subject: str
    body: str
    created_at: datetime
    category: Category | None = None
    urgency: Urgency | None = None
    sentiment: Sentiment | None = None
    ai_draft_reply: str | None = None
    ai_confidence: float | None = None
    confidence_flag: ConfidenceFlag | None = None
    human_reviewed: bool

    # Not a DB column: set only on the response from a request that just ran
    # the pipeline and had the LLM call fail, so the client knows the ticket
    # was saved but is not yet classified.
    classification_error: str | None = None


class TicketListResponse(BaseModel):
    items: list[TicketRead]
    total: int
    limit: int
    offset: int


class BatchImportRowResult(BaseModel):
    row: int
    status: Literal["created", "failed"]
    ticket_id: uuid.UUID | None = None
    error: str | None = None


class BatchImportResponse(BaseModel):
    total_rows: int
    created: int
    failed: int
    results: list[BatchImportRowResult]
