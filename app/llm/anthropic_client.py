import logging
import time

import anthropic
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from app.config import Settings, get_settings
from app.llm.base import (
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    TicketClassification,
)
from app.llm.prompts import CLASSIFY_TICKET_TOOL, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class AnthropicLLMClient:
    """LLMClient implementation backed by the Anthropic Messages API.

    Structurally satisfies `app.llm.base.LLMClient` (a Protocol) -- it is not
    a subclass of anything provider-specific outside this module.

    Only timeouts and rate limits are retried (with exponential backoff +
    jitter). Other failures -- bad API key, malformed request, 5xx -- are
    surfaced immediately as `LLMError` rather than retried, so persistent
    misconfiguration fails fast instead of being masked by retry attempts.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._model = settings.anthropic_model
        self._timeout = settings.llm_timeout_seconds
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_seconds
        )
        self._classify_with_retry = retry(
            reraise=True,
            stop=stop_after_attempt(max(settings.llm_max_retries, 1)),
            wait=wait_exponential_jitter(initial=1, max=10),
            retry=retry_if_exception_type((LLMTimeoutError, LLMRateLimitError)),
        )(self._classify_once)

    async def classify_ticket(self, subject: str, body: str) -> TicketClassification:
        return await self._classify_with_retry(subject, body)

    async def _classify_once(self, subject: str, body: str) -> TicketClassification:
        start = time.perf_counter()
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=[CLASSIFY_TICKET_TOOL],
                tool_choice={"type": "tool", "name": "classify_ticket"},
                messages=[
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(subject=subject, body=body),
                    }
                ],
            )
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(f"Anthropic API timed out after {self._timeout}s") from exc
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError("Anthropic API rate limit exceeded") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMTimeoutError(f"Could not connect to Anthropic API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(
                f"Anthropic API returned an error: {exc.status_code} {exc.message}"
            ) from exc
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API call failed: {exc}") from exc

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        classification = self._parse_response(response)

        logger.info(
            "llm classification completed",
            extra={
                "latency_ms": latency_ms,
                "input_tokens": classification.input_tokens,
                "output_tokens": classification.output_tokens,
                "model": self._model,
            },
        )
        return classification

    @staticmethod
    def _parse_response(response: anthropic.types.Message) -> TicketClassification:
        tool_use_block = next(
            (block for block in response.content if block.type == "tool_use"), None
        )
        if tool_use_block is None:
            raise LLMResponseError(
                "Anthropic response did not include a classify_ticket tool call"
            )

        try:
            return TicketClassification.model_validate(
                {
                    **tool_use_block.input,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            )
        except ValidationError as exc:
            raise LLMResponseError(
                f"Anthropic tool call output failed validation: {exc}"
            ) from exc
