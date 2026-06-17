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


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
