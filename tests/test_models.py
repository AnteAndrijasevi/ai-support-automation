import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.models import Ticket


async def test_ticket_has_sensible_defaults(db_session):
    ticket = Ticket(subject="Can't log in", body="Password reset isn't working.")
    db_session.add(ticket)
    await db_session.flush()

    assert isinstance(ticket.id, uuid.UUID)
    assert ticket.created_at is not None
    assert ticket.human_reviewed is False
    assert ticket.category is None
    assert ticket.confidence_flag is None


async def test_ticket_accepts_valid_category(db_session):
    ticket = Ticket(subject="s", body="b", category="billing", urgency="low", sentiment="neutral")
    db_session.add(ticket)
    await db_session.flush()
    assert ticket.category == "billing"


async def test_ticket_rejects_invalid_category(db_session):
    ticket = Ticket(subject="s", body="b", category="not_a_real_category")
    db_session.add(ticket)
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_ticket_rejects_invalid_confidence_flag(db_session):
    ticket = Ticket(subject="s", body="b", confidence_flag="bogus")
    db_session.add(ticket)
    with pytest.raises(IntegrityError):
        await db_session.flush()
