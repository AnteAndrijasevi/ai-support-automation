import enum
import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Category(enum.StrEnum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    GENERAL = "general"


class Urgency(enum.StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Sentiment(enum.StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ConfidenceFlag(enum.StrEnum):
    """Derived from the LLM's self-reported confidence score (see app/llm/base.py)."""

    OK = "ok"
    LOW_CONFIDENCE = "low_confidence"


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (
        CheckConstraint(
            f"category IN {tuple(c.value for c in Category)}", name="ck_tickets_category"
        ),
        CheckConstraint(f"urgency IN {tuple(u.value for u in Urgency)}", name="ck_tickets_urgency"),
        CheckConstraint(
            f"sentiment IN {tuple(s.value for s in Sentiment)}", name="ck_tickets_sentiment"
        ),
        CheckConstraint(
            f"confidence_flag IN {tuple(f.value for f in ConfidenceFlag)}",
            name="ck_tickets_confidence_flag",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    subject: Mapped[str] = mapped_column(String(255))
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    # Populated by the LLM pipeline after ingestion; null until processed.
    category: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    urgency: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    sentiment: Mapped[str | None] = mapped_column(String(16), nullable=True)
    ai_draft_reply: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_flag: Mapped[str | None] = mapped_column(String(16), nullable=True)

    human_reviewed: Mapped[bool] = mapped_column(default=False)
