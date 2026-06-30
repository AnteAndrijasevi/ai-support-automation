from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_support"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
