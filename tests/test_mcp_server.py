import json

from sqlalchemy import func, select

import app.mcp.server as mcp_server
from app.llm.base import TicketClassification
from app.mcp.server import mcp
from app.models import Ticket


async def test_mcp_exposes_classify_and_create_ticket_tools():
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert names == {"classify_ticket", "create_ticket"}


async def test_classify_ticket_tool_does_not_persist(monkeypatch, db_session_factory, fake_llm):
    fake_llm.response = TicketClassification(
        category="billing",
        urgency="high",
        sentiment="negative",
        confidence=0.9,
        draft_reply="Sorry about that -- we'll look into the duplicate charge.",
    )
    monkeypatch.setattr(mcp_server, "get_llm_client", lambda: fake_llm)

    result = await mcp.call_tool(
        "classify_ticket", {"subject": "Double charge", "body": "billed twice"}
    )

    payload = _extract_payload(result)
    assert payload["category"] == "billing"
    assert payload["draft_reply"].startswith("Sorry")
    assert "input_tokens" not in payload

    async with db_session_factory() as session:
        count = (await session.execute(select(func.count()).select_from(Ticket))).scalar_one()
        assert count == 0


async def test_create_ticket_tool_persists_ticket(monkeypatch, db_session_factory, fake_llm):
    fake_llm.response = TicketClassification(
        category="technical",
        urgency="medium",
        sentiment="neutral",
        confidence=0.8,
        draft_reply="We're looking into the sync issue.",
    )
    monkeypatch.setattr(mcp_server, "get_llm_client", lambda: fake_llm)
    monkeypatch.setattr(mcp_server, "async_session_factory", db_session_factory)

    result = await mcp.call_tool(
        "create_ticket", {"subject": "Sync broken", "body": "files won't sync"}
    )

    payload = _extract_payload(result)
    assert payload["category"] == "technical"
    assert payload["id"]

    async with db_session_factory() as session:
        count = (await session.execute(select(func.count()).select_from(Ticket))).scalar_one()
        assert count == 1


def _extract_payload(result) -> dict:
    """FastMCP returns (content_blocks, structured_dict) when structured output is
    inferred from the return type; normalize both shapes for easier assertions."""
    if isinstance(result, tuple):
        _, structured = result
        return structured
    if isinstance(result, dict):
        return result
    text = result[0].text
    return json.loads(text)
