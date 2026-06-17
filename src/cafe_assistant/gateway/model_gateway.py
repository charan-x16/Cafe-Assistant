from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

from cafe_assistant.config import settings


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


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


def _tokenize(text: str) -> Sequence[str]:
    return tuple(token for token in text.lower().split() if token)


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0:
        return vector
    return [component / magnitude for component in vector]
