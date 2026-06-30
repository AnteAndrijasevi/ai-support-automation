"""Orchestrates the ingest -> classify -> persist flow shared by every entry point
(REST API, batch import, MCP tool, eval harness)."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.base import LLMClient, LLMError, TicketClassification
from app.models import ConfidenceFlag, Ticket

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.6


def derive_confidence_flag(confidence: float) -> ConfidenceFlag:
    return (
        ConfidenceFlag.LOW_CONFIDENCE
        if confidence < LOW_CONFIDENCE_THRESHOLD
        else ConfidenceFlag.OK
    )


async def classify_ticket(llm_client: LLMClient, subject: str, body: str) -> TicketClassification:
    """Run a ticket through the LLM. Raises `LLMError` on failure; callers decide
    whether that should fail the request or persist the ticket unclassified."""
    return await llm_client.classify_ticket(subject=subject, body=body)


async def ingest_ticket(
    db: AsyncSession,
    llm_client: LLMClient,
    *,
    subject: str,
    body: str,
) -> Ticket:
    """Persist a new ticket and run it through the AI pipeline.

    The ticket row is always created. If the LLM call fails, the ticket is still
    saved (so no customer-submitted ticket is ever silently lost) but left
    unclassified for a later retry; the error is logged and re-raised so the
    caller (e.g. the API layer) can return an appropriate response.
    """
    ticket = Ticket(subject=subject, body=body)
    db.add(ticket)
    await db.flush()

    try:
        classification = await classify_ticket(llm_client, subject, body)
    except LLMError:
        await db.commit()
        logger.exception("LLM classification failed for ticket %s", ticket.id)
        raise

    ticket.category = classification.category
    ticket.urgency = classification.urgency
    ticket.sentiment = classification.sentiment
    ticket.ai_draft_reply = classification.draft_reply
    ticket.ai_confidence = classification.confidence
    ticket.confidence_flag = derive_confidence_flag(classification.confidence)

    await db.commit()
    await db.refresh(ticket)
    return ticket
