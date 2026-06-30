"""Covers the DB-failure error path: app.main.database_error_handler should turn
any SQLAlchemyError raised inside a route into a clean 503 instead of a raw 500
stack trace."""

from sqlalchemy.exc import OperationalError

from app.db import get_db
from app.main import app


class BrokenSession:
    """Stands in for a session whose underlying connection is unreachable."""

    async def execute(self, *args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    async def get(self, *args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))


async def test_list_tickets_returns_503_when_db_is_unreachable(client):
    async def _broken_get_db():
        yield BrokenSession()

    app.dependency_overrides[get_db] = _broken_get_db
    try:
        response = await client.get("/tickets")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 503
    assert response.json() == {"detail": "Database temporarily unavailable"}


async def test_get_ticket_returns_503_when_db_is_unreachable(client):
    async def _broken_get_db():
        yield BrokenSession()

    app.dependency_overrides[get_db] = _broken_get_db
    try:
        response = await client.get("/tickets/00000000-0000-0000-0000-000000000000")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 503
