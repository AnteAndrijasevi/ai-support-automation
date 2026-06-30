from functools import lru_cache

from app.llm.anthropic_client import AnthropicLLMClient
from app.llm.base import LLMClient


@lru_cache
def get_llm_client() -> LLMClient:
    """Returns the process-wide LLM client. Swapping providers means changing
    this one line -- everything else depends on the `LLMClient` protocol."""
    return AnthropicLLMClient()
