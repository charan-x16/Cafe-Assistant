"""Environment-backed configuration for the cafe assistant backend.

The application reads all runtime settings through this module so local Docker,
Neon Postgres, Qdrant, Redis, model providers, observability, and governance
settings share one typed source of truth. Validators normalize common deployment
URL formats into the async forms expected by the backend without leaking secrets
or requiring callers to know provider-specific quirks.
"""

from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables and `.env`.

    The fields in this class intentionally mirror deployment configuration: SQL,
    Redis, embeddings, Qdrant, chat providers, identity, rate limits, retention,
    tracing, and evaluation budgets. Defaults are safe for local development;
    production should override secrets, external service URLs, and provider keys.
    """

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
    qdrant_timeout_seconds: float = 3.0
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
    identity_otp_hash_secret: str = "local-dev-otp-hash-secret"
    identity_device_token_hash_secret: str = "local-dev-device-token-hash-secret"
    device_token_bytes: int = 32
    device_token_ttl_seconds: int = 60 * 60 * 24 * 90
    otp_code_ttl_seconds: int = 300
    otp_max_attempts: int = 5
    otp_store_provider: str = "memory"
    sms_provider: str = "noop"
    otp_allow_noop_sms_sender: bool = True
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
        """Normalize Postgres URLs into SQLAlchemy asyncpg-compatible URLs.

        Args:
            value (object):
                Raw `DATABASE_URL` value read from the environment. Non-string
                values are returned unchanged so Pydantic can report type errors.

        Returns:
            object:
                A normalized database URL string when `value` is a Postgres URL;
                otherwise the original value. The normalization converts
                `postgresql://` to `postgresql+asyncpg://`, maps `sslmode` to the
                asyncpg `ssl` query parameter, and removes unsupported
                `channel_binding` parameters commonly emitted by managed hosts.
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
        """Treat blank optional secret and endpoint values as missing.

        Args:
            value (object):
                Raw environment value for an optional key, URL, or endpoint field.

        Returns:
            object:
                `None` for blank strings, otherwise the original value. This lets
                callers use empty entries in `.env` without accidentally enabling
                a provider with an unusable empty credential.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _use_qdrant_endpoint_alias(self) -> "Settings":
        """Populate `qdrant_url` from the legacy `qdrant_endpoint` alias.

        Args:
            None.

        Returns:
            Settings:
                The same settings instance with `qdrant_url` filled from
                `qdrant_endpoint` when only the alias was provided.
        """
        if self.qdrant_url is None and self.qdrant_endpoint is not None:
            self.qdrant_url = self.qdrant_endpoint
        return self


@lru_cache
def get_settings() -> Settings:
    """Load and cache environment settings for the current process.

    Args:
        None.

    Returns:
        Settings:
            Cached typed settings object. The cache avoids repeatedly parsing
            `.env` and environment variables during request handling.
    """
    return Settings()


settings = get_settings()
