"""Implementation module for model gateway.
Contains typed helpers used by the cafe assistant backend runtime.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from cafe_assistant.config import settings


class EmbeddingProvider(Protocol):
    """Protocol that defines the embedding provider interface for swappable providers."""
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed the requested value.

        Args:
            texts (list[str]):
                Input texts that should each receive one embedding vector.

        Returns:
            list[list[float]]:
                Value produced for the caller according to the function contract.
        """


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Container for chat message behavior and data."""
    role: str
    content: str


class ChatProvider(Protocol):
    """Protocol that defines the chat provider interface for swappable providers."""
    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """Handle stream chat.

        Args:
            messages (list[ChatMessage]):
                Ordered chat messages sent to the configured chat provider.
            timeout_seconds (float):
                Maximum time allowed for the streaming chat request.

        Returns:
            AsyncIterator[str]:
                Streamed values yielded to the caller as they become available.
        """


class HashEmbeddingProvider:
    """Container for hash embedding provider behavior and data."""

    def __init__(self, dimensions: int) -> None:
        """Initialize the object with the dependencies or values required later.

        Args:
            dimensions (int):
                Expected embedding vector dimension count.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed the requested value.

        Args:
            texts (list[str]):
                Input texts that should each receive one embedding vector.

        Returns:
            list[list[float]]:
                Value produced for the caller according to the function contract.
        """
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Embed one.

        Args:
            text (str):
                Input text to normalize, embed, tokenize, or classify.

        Returns:
            list[float]:
                Value produced for the caller according to the function contract.
        """
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return _normalize(vector)


class SentenceTransformerEmbeddingProvider:
    """Container for sentence transformer embedding provider behavior and data."""

    def __init__(self, model_name: str, dimensions: int) -> None:
        """Initialize the object with the dependencies or values required later.

        Args:
            model_name (str):
                Provider model name used to create embeddings or chat responses.
            dimensions (int):
                Expected embedding vector dimension count.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
        self.model_name = model_name
        self.dimensions = dimensions
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "The sentence_transformer embedding provider requires the "
                "'sentence-transformers' package. Install project dependencies, then rerun "
                "the embedding backfill."
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed the requested value.

        Args:
            texts (list[str]):
                Input texts that should each receive one embedding vector.

        Returns:
            list[list[float]]:
                Value produced for the caller according to the function contract.
        """
        encoded = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vectors = [self._to_float_vector(vector) for vector in encoded]
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} dimensions from {self.model_name}, "
                    f"got {len(vector)}."
                )
        return vectors

    def _to_float_vector(self, vector: object) -> list[float]:
        """Convert float vector.

        Args:
            vector (object):
                Vector being normalized, converted, or sent to the vector store.

        Returns:
            list[float]:
                Value produced for the caller according to the function contract.
        """
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(component) for component in vector]  # type: ignore[union-attr]


class ConfiguredEmbeddingProvider:
    """Container for configured embedding provider behavior and data."""

    def __init__(self, provider_name: str | None = None, dimensions: int | None = None) -> None:
        """Initialize the object with the dependencies or values required later.

        Args:
            provider_name (str | None):
                Configured provider name used to select an adapter.
            dimensions (int | None):
                Expected embedding vector dimension count.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
        self.provider_name = provider_name or settings.embedding_provider
        self.dimensions = dimensions or settings.embedding_dimension
        self._provider = self._build_provider()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed the requested value.

        Args:
            texts (list[str]):
                Input texts that should each receive one embedding vector.

        Returns:
            list[list[float]]:
                Value produced for the caller according to the function contract.
        """
        return self._provider.embed(texts)

    def _build_provider(self) -> EmbeddingProvider:
        """Build provider.

        Args:
            None.

        Returns:
            EmbeddingProvider:
                Constructed value used by the caller for retrieval, tracing, or storage.
        """
        if self.provider_name == "hash":
            return HashEmbeddingProvider(self.dimensions)
        if self.provider_name in {
            "sentence_transformer",
            "sentence-transformer",
            "sentence_transformers",
        }:
            return SentenceTransformerEmbeddingProvider(
                settings.embedding_model_name,
                self.dimensions,
            )
        raise ValueError(f"Unsupported embedding provider: {self.provider_name}")


def get_embedding_provider() -> EmbeddingProvider:
    """Build the configured embedding provider for runtime use.

    Args:
        None.

    Returns:
        EmbeddingProvider:
            EmbeddingProvider selected from runtime settings.
    """
    return ConfiguredEmbeddingProvider()


class LocalChatProvider:
    """Container for local chat provider behavior and data."""

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """Handle stream chat.

        Args:
            messages (list[ChatMessage]):
                Ordered chat messages sent to the configured chat provider.
            timeout_seconds (float):
                Maximum time allowed for the streaming chat request.

        Returns:
            AsyncIterator[str]:
                Streamed values yielded to the caller as they become available.
        """
        del timeout_seconds
        prompt = messages[-1].content if messages else ""
        response = _local_response_from_prompt(prompt)
        for token in _chunk_text(response):
            yield token
            await asyncio.sleep(0)


class OpenAIChatProvider:
    """Container for open ai chat provider behavior and data."""

    def __init__(self, *, model_name: str, api_key: str | None) -> None:
        """Initialize the object with the dependencies or values required later.

        Args:
            model_name (str):
                Provider model name used to create embeddings or chat responses.
            api_key (str | None):
                Api key value required to perform this operation.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
        if not api_key:
            raise RuntimeError("LLM_API_KEY is required when using the OpenAI chat provider.")
        self.model_name = model_name
        self.api_key = api_key

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """Handle stream chat.

        Args:
            messages (list[ChatMessage]):
                Ordered chat messages sent to the configured chat provider.
            timeout_seconds (float):
                Maximum time allowed for the streaming chat request.

        Returns:
            AsyncIterator[str]:
                Streamed values yielded to the caller as they become available.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "stream": True,
            "temperature": 0.2,
        }
        timeout = httpx.Timeout(timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    token = _openai_token_from_sse_line(line)
                    if token is not None:
                        yield token


class FallbackChatProvider:
    """Container for fallback chat provider behavior and data."""
    def __init__(
        self,
        providers: list[ChatProvider],
        *,
        timeout_seconds: float,
        retries: int,
    ) -> None:
        """Initialize the object with the dependencies or values required later.

        Args:
            providers (list[ChatProvider]):
                Providers value required to perform this operation.
            timeout_seconds (float):
                Maximum time allowed for the streaming chat request.
            retries (int):
                Retries value required to perform this operation.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
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
        """Handle stream chat.

        Args:
            messages (list[ChatMessage]):
                Ordered chat messages sent to the configured chat provider.
            timeout_seconds (float | None):
                Maximum time allowed for the streaming chat request.

        Returns:
            AsyncIterator[str]:
                Streamed values yielded to the caller as they become available.
        """
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
    """Container for configured chat provider behavior and data."""
    def __init__(
        self,
        provider_name: str,
        *,
        timeout_seconds: float,
        retries: int,
    ) -> None:
        """Initialize the object with the dependencies or values required later.

        Args:
            provider_name (str):
                Configured provider name used to select an adapter.
            timeout_seconds (float):
                Maximum time allowed for the streaming chat request.
            retries (int):
                Retries value required to perform this operation.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
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
        """Handle stream chat.

        Args:
            messages (list[ChatMessage]):
                Ordered chat messages sent to the configured chat provider.
            timeout_seconds (float):
                Maximum time allowed for the streaming chat request.

        Returns:
            AsyncIterator[str]:
                Streamed values yielded to the caller as they become available.
        """
        async for token in self._provider.stream_chat(
            messages,
            timeout_seconds=timeout_seconds,
        ):
            yield token

    def _build_provider(self) -> ChatProvider:
        """Build provider.

        Args:
            None.

        Returns:
            ChatProvider:
                Constructed value used by the caller for retrieval, tracing, or storage.
        """
        if self.provider_name == "local":
            return LocalChatProvider()
        if self.provider_name == "openai":
            return OpenAIChatProvider(
                model_name=settings.llm_model,
                api_key=settings.llm_api_key,
            )
        raise ValueError(f"Unsupported chat provider: {self.provider_name}")


@dataclass(frozen=True, slots=True)
class ChatModelCascade:
    """Container for chat model cascade behavior and data."""
    cheap: ChatProvider
    strong: ChatProvider


def get_chat_model_cascade() -> ChatModelCascade:
    """Build the cheap/strong chat model cascade from configuration.

    Args:
        None.

    Returns:
        ChatModelCascade:
            Configured cheap and strong chat providers for the agent.
    """
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
    """Tokenize the requested value.

    Args:
        text (str):
            Input text to normalize, embed, tokenize, or classify.

    Returns:
        Sequence[str]:
            Value produced for the caller according to the function contract.
    """
    return tuple(token for token in text.lower().split() if token)


def _normalize(vector: list[float]) -> list[float]:
    """Normalize the requested value.

    Args:
        vector (list[float]):
            Vector being normalized, converted, or sent to the vector store.

    Returns:
        list[float]:
            Value produced for the caller according to the function contract.
    """
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0:
        return vector
    return [component / magnitude for component in vector]


def _local_response_from_prompt(prompt: str) -> str:
    """Handle local response from prompt.

    Args:
        prompt (str):
            Prompt value required to perform this operation.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
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


def _openai_token_from_sse_line(line: str) -> str | None:
    """Handle openai token from SSE line.

    Args:
        line (str):
            Single server-sent-events line returned by a chat provider.

    Returns:
        str | None:
            Value produced for the caller according to the function contract.
    """
    if not line.startswith("data: "):
        return None
    data = line.removeprefix("data: ").strip()
    if not data or data == "[DONE]":
        return None
    payload = json.loads(data)
    choices = payload.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) and content else None


def _chunk_text(text: str, chunk_size: int = 18) -> Sequence[str]:
    """Handle chunk text.

    Args:
        text (str):
            Input text to normalize, embed, tokenize, or classify.
        chunk_size (int):
            Chunk size value required to perform this operation.

    Returns:
        Sequence[str]:
            Value produced for the caller according to the function contract.
    """
    return tuple(text[index : index + chunk_size] for index in range(0, len(text), chunk_size))
