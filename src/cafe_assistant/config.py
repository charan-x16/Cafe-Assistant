"""Implementation module for config.
Contains typed helpers used by the cafe assistant backend runtime.
"""

from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Container for settings behavior and data."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Cafe Assistant"
    environment: str = "local"
    database_url: str = "postgresql+asyncpg://cafe:cafe@localhost:5432/cafe_assistant"
    redis_url: str = "redis://localhost:6379/0"
    embedding_provider: str = "sentence_transformer"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    vector_provider: str = "qdrant"
    qdrant_url: str | None = None
    qdrant_endpoint: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "cafe_assistant_menu_policy"
    llm_provider: str = "local"
    llm_model: str = "local"
    llm_api_key: str | None = None
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

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_database_url(cls, value: object) -> object:
        """Normalize database url.

        Args:
            value (object):
                Value value required to perform this operation.

        Returns:
            object:
                Value produced for the caller according to the function contract.
        """
        if not isinstance(value, str):
            return value
        database_url = value.strip().strip(chr(34)).strip(chr(39))
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        parts = urlsplit(database_url)
        if parts.scheme != "postgresql+asyncpg":
            return database_url

        query_pairs = parse_qsl(parts.query, keep_blank_values=True)
        normalized_pairs: list[tuple[str, str]] = []
        ssl_value: str | None = None
        for key, item_value in query_pairs:
            if key == "sslmode":
                if item_value and item_value != "disable":
                    ssl_value = item_value
                continue
            if key == "channel_binding":
                continue
            normalized_pairs.append((key, item_value))
        if ssl_value and not any(key == "ssl" for key, _ in normalized_pairs):
            normalized_pairs.append(("ssl", ssl_value))

        return urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                parts.path,
                urlencode(normalized_pairs),
                parts.fragment,
            )
        )

    @field_validator(
        "llm_api_key",
        "qdrant_url",
        "qdrant_endpoint",
        "qdrant_api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
        mode="before",
    )
    @classmethod
    def _blank_string_to_none(cls, value: object) -> object:
        """Handle blank string to none.

        Args:
            value (object):
                Value value required to perform this operation.

        Returns:
            object:
                Value produced for the caller according to the function contract.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _use_qdrant_endpoint_alias(self) -> "Settings":
        """Handle use Qdrant endpoint alias.

        Args:
            None.

        Returns:
            'Settings':
                Value produced for the caller according to the function contract.
        """
        if self.qdrant_url is None and self.qdrant_endpoint is not None:
            self.qdrant_url = self.qdrant_endpoint
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the cached environment settings object.

    Args:
        None.

    Returns:
        Settings:
            Cached Settings instance loaded from environment variables.
    """
    return Settings()


settings = get_settings()