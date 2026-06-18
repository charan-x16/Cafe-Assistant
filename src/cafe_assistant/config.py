from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Cafe Assistant"
    environment: str = "local"
    database_url: str = "postgresql+asyncpg://cafe:cafe@localhost:5432/cafe_assistant"
    redis_url: str = "redis://localhost:6379/0"
    embedding_provider: str = "hash"
    embedding_dimension: int = 8
    cheap_chat_provider: str = "local"
    strong_chat_provider: str = "local"
    chat_timeout_seconds: float = 8.0
    chat_retries: int = 1
    agent_deadline_seconds: float = 12.0
    agent_max_tool_calls: int = 4
    identity_phone_hash_secret: str = "local-dev-phone-hash-secret"
    device_token_bytes: int = 32
    otp_code_ttl_seconds: int = 300
    rate_limit_session_requests: int = 60
    rate_limit_session_window_seconds: int = 60
    rate_limit_ip_requests: int = 120
    rate_limit_ip_window_seconds: int = 60
    profile_retention_days: int = 365
    session_retention_days: int = 14
    audit_retention_days: int = 730
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    default_chat_model_name: str = "local"
    cheap_model_input_cost_per_1k: float = 0.0
    cheap_model_output_cost_per_1k: float = 0.0
    strong_model_input_cost_per_1k: float = 0.0
    strong_model_output_cost_per_1k: float = 0.0
    latency_budget_ms: float = 1500.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
