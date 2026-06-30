import pytest

from app.llm.base import LLMTimeoutError, TicketClassification
from app.models import ConfidenceFlag
from app.services.ticket_pipeline import derive_confidence_flag, ingest_ticket
from tests.fakes import FakeLLMClient


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.95, ConfidenceFlag.OK),
        (0.6, ConfidenceFlag.OK),
        (0.59, ConfidenceFlag.LOW_CONFIDENCE),
        (0.0, ConfidenceFlag.LOW_CONFIDENCE),
    ],
)
def test_derive_confidence_flag(confidence, expected):
    assert derive_confidence_flag(confidence) == expected


async def test_ingest_ticket_persists_classification_on_success(db_session):
    fake = FakeLLMClient(
        response=TicketClassification(
            category="technical",
            urgency="medium",
            sentiment="neutral",
            confidence=0.8,
            draft_reply="Thanks, we're looking into the sync issue.",
            input_tokens=10,
            output_tokens=10,
        )
    )

    result = await ingest_ticket(db_session, fake, subject="Sync broken", body="Files won't sync.")

    assert result.classification_error is None
    assert result.ticket.category == "technical"
    assert result.ticket.confidence_flag == ConfidenceFlag.OK
    assert fake.calls == [("Sync broken", "Files won't sync.")]


async def test_ingest_ticket_flags_low_confidence(db_session):
    fake = FakeLLMClient(
        response=TicketClassification(
            category="general",
            urgency="low",
            sentiment="neutral",
            confidence=0.3,
            draft_reply="Could you clarify what you mean?",
        )
    )

    result = await ingest_ticket(db_session, fake, subject="Vague", body="It's broken I guess?")

    assert result.ticket.confidence_flag == ConfidenceFlag.LOW_CONFIDENCE


async def test_ingest_ticket_survives_llm_failure(db_session):
    fake = FakeLLMClient(error=LLMTimeoutError("timed out"))

    result = await ingest_ticket(db_session, fake, subject="Help", body="Something is wrong.")

    assert result.classification_error == "timed out"
    assert result.ticket.id is not None
    assert result.ticket.category is None
    assert result.ticket.confidence_flag is None
