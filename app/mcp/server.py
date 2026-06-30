"""Exposes ticket-automation functions as MCP tools, alongside (not instead of) the
REST API. Both entry points call the exact same `app.services.ticket_pipeline`
functions, so an MCP client (e.g. an AI assistant wired up to this server) and the
REST API always see identical classification behavior.

Run with: python -m app.mcp.server
"""

from mcp.server.fastmcp import FastMCP

from app.db import async_session_factory
from app.dependencies import get_llm_client
from app.services.ticket_pipeline import classify_ticket as run_classification
from app.services.ticket_pipeline import ingest_ticket

mcp = FastMCP(
    "ai-support-automation",
    instructions=(
        "Tools for triaging customer support tickets: classifying a ticket's "
        "category/urgency/sentiment and drafting a reply, or doing that and "
        "persisting the ticket to the database."
    ),
)


@mcp.tool()
async def classify_ticket(subject: str, body: str) -> dict:
    """Classify a support ticket and draft a reply, without saving it anywhere.

    Useful for previewing how a ticket would be triaged. Returns category,
    urgency, sentiment, a confidence score, and a drafted reply.
    """
    llm_client = get_llm_client()
    result = await run_classification(llm_client, subject, body)
    return result.model_dump(exclude={"input_tokens", "output_tokens"})


@mcp.tool()
async def create_ticket(subject: str, body: str) -> dict:
    """Ingest a support ticket: persist it and run it through the AI triage pipeline.

    Equivalent to POST /tickets on the REST API. Returns the saved ticket,
    including its id, so it can be looked up later via GET /tickets/{id}.
    """
    llm_client = get_llm_client()
    async with async_session_factory() as session:
        result = await ingest_ticket(session, llm_client, subject=subject, body=body)

    return {
        "id": str(result.ticket.id),
        "subject": result.ticket.subject,
        "category": result.ticket.category,
        "urgency": result.ticket.urgency,
        "sentiment": result.ticket.sentiment,
        "ai_draft_reply": result.ticket.ai_draft_reply,
        "confidence_flag": result.ticket.confidence_flag,
        "classification_error": result.classification_error,
    }


if __name__ == "__main__":
    mcp.run()
