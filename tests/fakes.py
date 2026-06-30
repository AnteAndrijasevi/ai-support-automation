"""Test doubles for the LLMClient protocol.

Because LLMClient is a structural Protocol, a fake only needs to implement
classify_ticket -- no base class, no mocking framework required.
"""

from app.llm.base import TicketClassification

DEFAULT_CLASSIFICATION = TicketClassification(
    category="general",
    urgency="low",
    sentiment="neutral",
    confidence=0.9,
    draft_reply="Thanks for reaching out -- we'll look into this and follow up shortly.",
    input_tokens=50,
    output_tokens=20,
)


class FakeLLMClient:
    def __init__(
        self,
        response: TicketClassification | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or DEFAULT_CLASSIFICATION
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def classify_ticket(self, subject: str, body: str) -> TicketClassification:
        self.calls.append((subject, body))
        if self.error is not None:
            raise self.error
        return self.response
