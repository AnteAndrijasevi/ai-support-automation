"""Orchestrates the ingest -> classify -> persist flow shared by every entry point
(REST API, batch import, MCP tool, eval harness)."""

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.base import LLMClient, LLMError, TicketClassification
from app.models import ConfidenceFlag, Ticket

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.6


@dataclass
class IngestResult:
    ticket: Ticket
    classification_error: str | None = None


def derive_confidence_flag(confidence: float) -> ConfidenceFlag:
    return (
        ConfidenceFlag.LOW_CONFIDENCE
        if confidence < LOW_CONFIDENCE_THRESHOLD
        else ConfidenceFlag.OK
    )


async def classify_ticket(llm_client: LLMClient, subject: str, body: str) -> TicketClassification:
    """Run a ticket through the LLM. Raises `LLMError` on failure."""
    return await llm_client.classify_ticket(subject=subject, body=body)


async def ingest_ticket(
    db: AsyncSession,
    llm_client: LLMClient,
    *,
    subject: str,
    body: str,
) -> IngestResult:
    """Persist a new ticket and run it through the AI pipeline.

    The ticket row is always created and committed -- no customer-submitted
    ticket is ever lost because the LLM was unavailable. If classification
    fails, the ticket is saved unclassified and the result carries a
    human-readable `classification_error` instead of raising, so callers
    (API routes, batch import, the MCP tool) all get the same degrade-gracefully
    behavior without each having to special-case `LLMError`.
    """
    ticket = Ticket(subject=subject, body=body)
    db.add(ticket)
    await db.flush()

    try:
        classification = await classify_ticket(llm_client, subject, body)
    except LLMError as exc:
        await db.commit()
        logger.warning("LLM classification failed for ticket %s: %s", ticket.id, exc)
        return IngestResult(ticket=ticket, classification_error=str(exc))

    ticket.category = classification.category
    ticket.urgency = classification.urgency
    ticket.sentiment = classification.sentiment
    ticket.ai_draft_reply = classification.draft_reply
    ticket.ai_confidence = classification.confidence
    ticket.confidence_flag = derive_confidence_flag(classification.confidence)

    await db.commit()
    await db.refresh(ticket)
    return IngestResult(ticket=ticket)
