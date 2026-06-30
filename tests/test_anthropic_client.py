from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic
import httpx
import pytest

from app.config import Settings
from app.llm.anthropic_client import AnthropicLLMClient
from app.llm.base import LLMRateLimitError, LLMResponseError, LLMTimeoutError


def make_settings(**overrides) -> Settings:
    defaults = {
        "anthropic_api_key": "sk-ant-test-key",
        "anthropic_model": "claude-sonnet-4-5",
        "llm_max_retries": 3,
        "llm_timeout_seconds": 5,
    }
    return Settings(**{**defaults, **overrides})


def make_tool_use_response(
    *,
    category="billing",
    urgency="high",
    sentiment="negative",
    confidence=0.85,
    draft_reply="We're looking into this now.",
    input_tokens=100,
    output_tokens=30,
):
    tool_block = SimpleNamespace(
        type="tool_use",
        input={
            "category": category,
            "urgency": urgency,
            "sentiment": sentiment,
            "confidence": confidence,
            "draft_reply": draft_reply,
        },
    )
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[tool_block], usage=usage)


def make_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def make_api_status_error(status_code: int, message: str = "error") -> anthropic.APIStatusError:
    return anthropic.APIStatusError(
        message, response=httpx.Response(status_code, request=make_request()), body=None
    )


def make_rate_limit_error(message: str = "rate limited") -> anthropic.RateLimitError:
    return anthropic.RateLimitError(
        message, response=httpx.Response(429, request=make_request()), body=None
    )


@pytest.fixture
def llm_client():
    client = AnthropicLLMClient(settings=make_settings())
    client._client.messages.create = AsyncMock()
    return client


async def test_classify_ticket_parses_valid_tool_response(llm_client):
    llm_client._client.messages.create.return_value = make_tool_use_response()

    result = await llm_client.classify_ticket("Billing issue", "I was charged twice.")

    assert result.category == "billing"
    assert result.urgency == "high"
    assert result.sentiment == "negative"
    assert result.confidence == 0.85
    assert result.input_tokens == 100
    assert result.output_tokens == 30
    llm_client._client.messages.create.assert_awaited_once()


async def test_classify_ticket_raises_on_missing_tool_use_block(llm_client):
    llm_client._client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="I cannot help with tool calls.")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )

    with pytest.raises(LLMResponseError):
        await llm_client.classify_ticket("s", "b")

    llm_client._client.messages.create.assert_awaited_once()


async def test_classify_ticket_raises_on_invalid_tool_input(llm_client):
    bad_response = make_tool_use_response()
    bad_response.content[0].input = {"category": "not_a_category"}
    llm_client._client.messages.create.return_value = bad_response

    with pytest.raises(LLMResponseError):
        await llm_client.classify_ticket("s", "b")

    # Validation failures are not retried.
    llm_client._client.messages.create.assert_awaited_once()


async def test_classify_ticket_retries_timeout_then_succeeds(llm_client):
    llm_client._client.messages.create.side_effect = [
        anthropic.APITimeoutError(make_request()),
        make_tool_use_response(),
    ]

    result = await llm_client.classify_ticket("s", "b")

    assert result.category == "billing"
    assert llm_client._client.messages.create.await_count == 2


async def test_classify_ticket_retries_rate_limit_until_attempts_exhausted(llm_client):
    llm_client._client.messages.create.side_effect = make_rate_limit_error()

    with pytest.raises(LLMRateLimitError):
        await llm_client.classify_ticket("s", "b")

    assert llm_client._client.messages.create.await_count == 3  # llm_max_retries


async def test_classify_ticket_does_not_retry_bad_request(llm_client):
    llm_client._client.messages.create.side_effect = make_api_status_error(400, "bad request")

    with pytest.raises(Exception, match="bad request"):
        await llm_client.classify_ticket("s", "b")

    llm_client._client.messages.create.assert_awaited_once()


async def test_classify_ticket_wraps_connection_error_as_timeout(llm_client):
    llm_client._client.messages.create.side_effect = anthropic.APIConnectionError(
        request=make_request()
    )

    with pytest.raises(LLMTimeoutError):
        await llm_client.classify_ticket("s", "b")

    assert llm_client._client.messages.create.await_count == 3
