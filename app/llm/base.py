"""Provider-agnostic LLM contract for ticket triage.

`LLMClient` is a `typing.Protocol`, not an ABC: any class with a matching
`classify_ticket` coroutine satisfies it, so adding a second provider (e.g. a
local model or a different vendor) means writing one new class -- nothing in
`app/services` or `app/api` needs to change, and tests can satisfy the
protocol with a plain stub instead of subclassing anything.
"""

from typing import Protocol

from pydantic import BaseModel, Field

from app.models import Category, Sentiment, Urgency


class TicketClassification(BaseModel):
    """Structured output of a single ticket triage call."""

    category: Category
    urgency: Urgency
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0, description="Model's self-rated confidence, 0-1.")
    draft_reply: str = Field(min_length=1, description="Suggested reply to send the customer.")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class LLMError(Exception):
    """Base class for all LLM-adapter errors. Callers can catch just this."""


class LLMTimeoutError(LLMError):
    """The provider did not respond within the configured timeout."""


class LLMRateLimitError(LLMError):
    """The provider rejected the request due to rate limiting."""


class LLMResponseError(LLMError):
    """The provider responded, but the response could not be parsed/validated."""


class LLMClient(Protocol):
    async def classify_ticket(self, subject: str, body: str) -> TicketClassification:
        """Classify a ticket and draft a reply. Raises an `LLMError` subclass on failure."""
        ...
