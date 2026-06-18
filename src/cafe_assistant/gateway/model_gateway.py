from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from cafe_assistant.config import settings


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


class ChatProvider(Protocol):
    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """Stream chat tokens for a message list."""


class HashEmbeddingProvider:
    """Deterministic local embedding provider for development and tests."""

    def __init__(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return _normalize(vector)


class ConfiguredEmbeddingProvider:
    """Provider-agnostic adapter selected by environment configuration."""

    def __init__(self, provider_name: str | None = None, dimensions: int | None = None) -> None:
        self.provider_name = provider_name or settings.embedding_provider
        self.dimensions = dimensions or settings.embedding_dimension
        self._provider = self._build_provider()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed(texts)

    def _build_provider(self) -> EmbeddingProvider:
        if self.provider_name == "hash":
            return HashEmbeddingProvider(self.dimensions)
        raise ValueError(f"Unsupported embedding provider: {self.provider_name}")


def get_embedding_provider() -> EmbeddingProvider:
    return ConfiguredEmbeddingProvider()


class LocalChatProvider:
    """Deterministic local chat provider used until a real provider is configured."""

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        del timeout_seconds
        prompt = messages[-1].content if messages else ""
        response = _local_response_from_prompt(prompt)
        for token in _chunk_text(response):
            yield token
            await asyncio.sleep(0)


class FallbackChatProvider:
    def __init__(
        self,
        providers: list[ChatProvider],
        *,
        timeout_seconds: float,
        retries: int,
    ) -> None:
        if not providers:
            raise ValueError("At least one chat provider is required.")
        self.providers = providers
        self.timeout_seconds = timeout_seconds
        self.retries = max(retries, 0)

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[str]:
        timeout = timeout_seconds or self.timeout_seconds
        last_error: Exception | None = None
        for provider in self.providers:
            for _attempt in range(self.retries + 1):
                try:
                    async with asyncio.timeout(timeout):
                        async for token in provider.stream_chat(
                            messages,
                            timeout_seconds=timeout,
                        ):
                            yield token
                    return
                except Exception as exc:  # noqa: BLE001 - fallback should catch provider failures.
                    last_error = exc
                    continue
        if last_error is not None:
            raise last_error


class ConfiguredChatProvider:
    def __init__(
        self,
        provider_name: str,
        *,
        timeout_seconds: float,
        retries: int,
    ) -> None:
        self.provider_name = provider_name
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self._provider = FallbackChatProvider(
            [self._build_provider(), LocalChatProvider()],
            timeout_seconds=timeout_seconds,
            retries=retries,
        )

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        async for token in self._provider.stream_chat(
            messages,
            timeout_seconds=timeout_seconds,
        ):
            yield token

    def _build_provider(self) -> ChatProvider:
        if self.provider_name == "local":
            return LocalChatProvider()
        raise ValueError(f"Unsupported chat provider: {self.provider_name}")


@dataclass(frozen=True, slots=True)
class ChatModelCascade:
    cheap: ChatProvider
    strong: ChatProvider


def get_chat_model_cascade() -> ChatModelCascade:
    timeout = settings.chat_timeout_seconds
    retries = settings.chat_retries
    return ChatModelCascade(
        cheap=ConfiguredChatProvider(
            settings.cheap_chat_provider,
            timeout_seconds=timeout,
            retries=retries,
        ),
        strong=ConfiguredChatProvider(
            settings.strong_chat_provider,
            timeout_seconds=timeout,
            retries=retries,
        ),
    )


def _tokenize(text: str) -> Sequence[str]:
    return tuple(token for token in text.lower().split() if token)


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0:
        return vector
    return [component / magnitude for component in vector]


def _local_response_from_prompt(prompt: str) -> str:
    item_names = [
        line.split(":", 1)[1].split("|", 1)[0].strip()
        for line in prompt.splitlines()
        if line.startswith("SAFE_ITEM:")
    ]
    if item_names:
        if len(item_names) == 1:
            return f"I can suggest {item_names[0]}."
        return f"I can suggest {', '.join(item_names[:-1])}, or {item_names[-1]}."
    return "I can help with the cafe menu, but I need a menu item or preference to check."


def _chunk_text(text: str, chunk_size: int = 18) -> Sequence[str]:
    return tuple(text[index : index + chunk_size] for index in range(0, len(text), chunk_size))
