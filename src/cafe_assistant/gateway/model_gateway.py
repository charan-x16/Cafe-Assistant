"""Provider-neutral model gateway for embeddings and streaming chat.

This module is the boundary between the cafe assistant and external model
providers. It owns provider selection, retries, fallback, timeout accounting,
streaming semantics, and usage/cost metadata. The agent is still responsible
for deciding which safe menu context can reach a model.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import httpx

from cafe_assistant.config import settings
from cafe_assistant.observability.metrics import record_quality_event
from cafe_assistant.observability.tracing import estimate_cost, span, token_count


class ModelGatewayError(RuntimeError):
    """Base error for model gateway failures."""


class ModelProviderConfigError(ModelGatewayError):
    """Raised for invalid provider configuration or permanent request shape errors."""


class ModelProviderAuthError(ModelGatewayError):
    """Raised when a provider rejects the configured credentials."""


class ModelProviderTransientError(ModelGatewayError):
    """Raised for provider failures that may be retried or sent to fallback."""


class ModelProviderTimeoutError(ModelProviderTransientError):
    """Raised when provider work exceeds the remaining gateway deadline."""


@dataclass(frozen=True, slots=True)
class ChatProviderDescriptor:
    """Static provider metadata used for traces and cost estimates.

    Args:
        provider_name (str): Provider adapter name, such as `openai` or `local`.
        model_name (str): Model name sent to or represented by the provider.
        input_cost_per_1k (float): Prompt-token cost in USD per thousand tokens.
        output_cost_per_1k (float): Completion-token cost in USD per thousand tokens.
    """

    provider_name: str
    model_name: str
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0


@dataclass(frozen=True, slots=True)
class ChatUsage:
    """Provider-reported token usage when an API exposes it.

    Args:
        input_tokens (int | None): Provider prompt-token count.
        output_tokens (int | None): Provider completion-token count.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ChatCallMetadata:
    """Metadata from the most recent gateway-managed chat call.

    Args:
        provider_name (str): Actual provider that completed the response.
        model_name (str): Actual model that completed the response.
        configured_provider_name (str): Primary provider selected by configuration.
        input_tokens (int): Prompt tokens used for usage/cost accounting.
        output_tokens (int): Completion tokens used for usage/cost accounting.
        estimated_cost_usd (float): Estimated spend for the call.
        attempts (int): Provider attempts made before success.
        retry_count (int): Failed attempts before success.
        fallback_used (bool): Whether a non-primary provider completed the call.
        timeout_seconds (float): Total deadline passed to the gateway call.
    """

    provider_name: str
    model_name: str
    configured_provider_name: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    attempts: int
    retry_count: int
    fallback_used: bool
    timeout_seconds: float


class EmbeddingProvider(Protocol):
    """Synchronous embedding provider interface used by retrieval and backfill code."""
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in input order.

        Args:
            texts (list[str]):
                Text values that should each receive one embedding vector.

        Returns:
            list[list[float]]:
                One embedding vector per input text, preserving input order.
        """


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One guarded message sent to a chat provider."""
    role: str
    content: str


class ChatProvider(Protocol):
    """Asynchronous streaming chat provider interface used by the agent."""
    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """Stream a provider response as text chunks.

        Args:
            messages (list[ChatMessage]):
                Complete guarded prompt messages for this model call.
            timeout_seconds (float):
                Remaining deadline budget for the provider call.

        Returns:
            AsyncIterator[str]:
                Text chunks emitted in provider order.
        """


class HashEmbeddingProvider:
    """Deterministic local embedding provider for tests and offline use."""

    def __init__(self, dimensions: int) -> None:
        """Create the provider with the dependencies required for later calls.

        Args:
            dimensions (int):
                Expected embedding vector dimension count.

        Returns:
            None:
                No value is returned; the provider stores validated runtime configuration.
        """
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in input order.

        Args:
            texts (list[str]):
                Text values that should each receive one embedding vector.

        Returns:
            list[list[float]]:
                One embedding vector per input text, preserving input order.
        """
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Embed one.

        Args:
            text (str):
                Text to tokenize or embed.

        Returns:
            list[float]:
                Plain or normalized float vector for the supplied value.
        """
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return _normalize(vector)


class SentenceTransformerEmbeddingProvider:
    """Sentence Transformers adapter for local BGE embeddings."""

    def __init__(self, model_name: str, dimensions: int) -> None:
        """Create the provider with the dependencies required for later calls.

        Args:
            model_name (str):
                Provider model name used to create embeddings or chat responses.
            dimensions (int):
                Expected embedding vector dimension count.

        Returns:
            None:
                No value is returned; the provider stores validated runtime configuration.
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
        """Embed texts in input order.

        Args:
            texts (list[str]):
                Text values that should each receive one embedding vector.

        Returns:
            list[list[float]]:
                One embedding vector per input text, preserving input order.
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
        """Convert a provider vector object into plain Python floats.

        Args:
            vector (object):
                Array-like vector returned by the embedding provider.

        Returns:
            list[float]:
                Plain or normalized float vector for the supplied value.
        """
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(component) for component in vector]  # type: ignore[union-attr]


class ConfiguredEmbeddingProvider:
    """Embedding provider selected from environment-backed settings."""

    def __init__(self, provider_name: str | None = None, dimensions: int | None = None) -> None:
        """Create the provider with the dependencies required for later calls.

        Args:
            provider_name (str | None):
                Configured provider name used to select an adapter.
            dimensions (int | None):
                Expected embedding vector dimension count.

        Returns:
            None:
                No value is returned; the provider stores validated runtime configuration.
        """
        self.provider_name = provider_name or settings.embedding_provider
        self.dimensions = dimensions or settings.embedding_dimension
        self._provider = self._build_provider()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in input order.

        Args:
            texts (list[str]):
                Text values that should each receive one embedding vector.

        Returns:
            list[list[float]]:
                One embedding vector per input text, preserving input order.
        """
        return self._provider.embed(texts)

    def _build_provider(self) -> EmbeddingProvider:
        """Create the concrete provider selected by this instance.

        Args:
            None.

        Returns:
            EmbeddingProvider:
                Concrete provider used by retrieval, tracing, or chat calls.
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


@lru_cache(maxsize=1)
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
    """Deterministic no-network chat provider used for tests and fallback."""

    def __init__(self, *, model_name: str = "local") -> None:
        """Create a local provider.

        Args:
            model_name (str): Trace label for the local provider.

        Returns:
            None: Descriptor metadata is stored on the provider.
        """
        self.descriptor = ChatProviderDescriptor("local", model_name)

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """Stream deterministic text from `SAFE_ITEM` prompt lines.

        Args:
            messages (list[ChatMessage]): Guarded prompt messages.
            timeout_seconds (float): Accepted for protocol compatibility.

        Returns:
            AsyncIterator[str]: Local response chunks.
        """
        del timeout_seconds
        prompt = messages[-1].content if messages else ""
        response = _local_response_from_prompt(prompt)
        for token in _chunk_text(response):
            yield token
            await asyncio.sleep(0)


class OpenAIChatProvider:
    """OpenAI Chat Completions streaming adapter."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None,
        input_cost_per_1k: float = 0.0,
        output_cost_per_1k: float = 0.0,
    ) -> None:
        """Create an OpenAI chat provider.

        Args:
            model_name (str): OpenAI chat model name.
            api_key (str | None): API key used for Authorization.
            input_cost_per_1k (float): Prompt-token cost estimate.
            output_cost_per_1k (float): Completion-token cost estimate.

        Returns:
            None: Credentials and descriptor metadata are stored.
        """
        if not api_key:
            raise ModelProviderConfigError("LLM_API_KEY is required for OpenAI chat.")
        self.model_name = model_name
        self.api_key = api_key
        self.descriptor = ChatProviderDescriptor(
            "openai",
            model_name,
            input_cost_per_1k=input_cost_per_1k,
            output_cost_per_1k=output_cost_per_1k,
        )
        self.last_usage: ChatUsage | None = None

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """Stream text deltas from OpenAI.

        Args:
            messages (list[ChatMessage]): Guarded prompt messages.
            timeout_seconds (float): Remaining HTTP deadline.

        Returns:
            AsyncIterator[str]: Text chunks emitted by OpenAI.
        """
        self.last_usage = None
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.2,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise _openai_status_error(exc) from exc
                    async for line in response.aiter_lines():
                        chunk = _openai_payload_from_sse_line(line)
                        if chunk is None:
                            continue
                        usage = _openai_usage_from_payload(chunk)
                        if usage is not None:
                            self.last_usage = usage
                        token = _openai_token_from_payload(chunk)
                        if token is not None:
                            yield token
        except httpx.TimeoutException as exc:
            raise ModelProviderTimeoutError("OpenAI request timed out.") from exc
        except httpx.TransportError as exc:
            raise ModelProviderTransientError("OpenAI transport failed.") from exc


class FallbackChatProvider:
    """Retry and fallback wrapper that avoids mixed partial responses."""

    def __init__(
        self,
        providers: list[ChatProvider],
        *,
        timeout_seconds: float,
        retries: int,
        configured_provider_name: str = "unknown",
    ) -> None:
        """Create a fallback provider chain.

        Args:
            providers (list[ChatProvider]): Ordered primary and fallback providers.
            timeout_seconds (float): Default total deadline across all attempts.
            retries (int): Retry count per provider for retryable failures.
            configured_provider_name (str): Primary provider selected by config.

        Returns:
            None: Retry policy and providers are stored.
        """
        if not providers:
            raise ValueError("At least one chat provider is required.")
        self.providers = providers
        self.timeout_seconds = timeout_seconds
        self.retries = max(retries, 0)
        self.configured_provider_name = configured_provider_name
        self.last_metadata: ChatCallMetadata | None = None

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream chunks from the first provider that completes successfully.

        Args:
            messages (list[ChatMessage]): Guarded prompt messages.
            timeout_seconds (float | None): Optional total deadline override.

        Returns:
            AsyncIterator[str]: Complete chunks from one successful provider only.
        """
        chunks = await self._collect_with_fallback(
            messages,
            timeout_seconds=timeout_seconds or self.timeout_seconds,
        )
        for chunk in chunks:
            yield chunk

    async def _collect_with_fallback(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> list[str]:
        """Collect a complete response with one global deadline.

        Args:
            messages (list[ChatMessage]): Guarded prompt messages.
            timeout_seconds (float): Total deadline for all attempts.

        Returns:
            list[str]: Complete response chunks from the successful provider.
        """
        self.last_metadata = None
        deadline_at = asyncio.get_running_loop().time() + timeout_seconds
        input_tokens = token_count("\n".join(message.content for message in messages))
        attempts = 0
        last_error: Exception | None = None
        for provider_index, provider in enumerate(self.providers):
            descriptor = describe_chat_provider(provider)
            for attempt_index in range(self.retries + 1):
                attempts += 1
                remaining = deadline_at - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise ModelProviderTimeoutError("Chat provider deadline exceeded.")
                try:
                    chunks = await self._collect_one_attempt(
                        provider,
                        messages,
                        descriptor=descriptor,
                        timeout_seconds=remaining,
                        attempt_index=attempt_index,
                        provider_index=provider_index,
                        input_tokens=input_tokens,
                    )
                except Exception as exc:  # noqa: BLE001 - the gateway owns fallback policy.
                    last_error = exc
                    retryable = _is_retryable_exception(exc)
                    record_quality_event(
                        "llm_provider_failures_total",
                        provider=descriptor.provider_name,
                        model=descriptor.model_name,
                        retryable=str(retryable).lower(),
                    )
                    if not retryable:
                        raise
                    continue
                metadata = self._metadata_for_success(
                    provider,
                    descriptor=descriptor,
                    chunks=chunks,
                    input_tokens=input_tokens,
                    attempts=attempts,
                    provider_index=provider_index,
                    timeout_seconds=timeout_seconds,
                )
                self.last_metadata = metadata
                if metadata.fallback_used:
                    record_quality_event(
                        "llm_provider_fallback_total",
                        provider=self.configured_provider_name,
                        fallback_provider=metadata.provider_name,
                    )
                return chunks
        if last_error is not None:
            raise last_error
        raise ModelProviderTransientError("No chat provider produced a response.")

    async def _collect_one_attempt(
        self,
        provider: ChatProvider,
        messages: list[ChatMessage],
        *,
        descriptor: ChatProviderDescriptor,
        timeout_seconds: float,
        attempt_index: int,
        provider_index: int,
        input_tokens: int,
    ) -> list[str]:
        """Collect one provider attempt before yielding anything to the caller.

        Args:
            provider (ChatProvider): Provider being attempted.
            messages (list[ChatMessage]): Guarded prompt messages.
            descriptor (ChatProviderDescriptor): Provider metadata for spans.
            timeout_seconds (float): Remaining deadline for this attempt.
            attempt_index (int): Zero-based attempt index for this provider.
            provider_index (int): Zero-based index in the provider chain.
            input_tokens (int): Estimated prompt tokens for tracing.

        Returns:
            list[str]: Complete response chunks if the attempt succeeds.
        """
        chunks: list[str] = []
        with span(
            "llm.provider_attempt",
            provider=descriptor.provider_name,
            model=descriptor.model_name,
            configured_provider=self.configured_provider_name,
            attempt=attempt_index + 1,
            fallback_candidate=provider_index > 0,
            timeout_seconds=round(timeout_seconds, 3),
            input_tokens=input_tokens,
        ) as record:
            try:
                async with asyncio.timeout(timeout_seconds):
                    async for token in provider.stream_chat(
                        messages,
                        timeout_seconds=timeout_seconds,
                    ):
                        chunks.append(token)
            except TimeoutError as exc:
                record.attributes.update(
                    {"success": False, "error_type": "TimeoutError", "retryable": True}
                )
                raise ModelProviderTimeoutError("Chat provider attempt timed out.") from exc
            except Exception as exc:
                retryable = _is_retryable_exception(exc)
                record.attributes.update(
                    {"success": False, "error_type": type(exc).__name__, "retryable": retryable}
                )
                raise
            record.attributes.update(
                {
                    "success": True,
                    "output_tokens": token_count("".join(chunks)),
                    "chunk_count": len(chunks),
                }
            )
            return chunks

    def _metadata_for_success(
        self,
        provider: ChatProvider,
        *,
        descriptor: ChatProviderDescriptor,
        chunks: list[str],
        input_tokens: int,
        attempts: int,
        provider_index: int,
        timeout_seconds: float,
    ) -> ChatCallMetadata:
        """Build metadata for a successful provider response.

        Args:
            provider (ChatProvider): Provider that completed the call.
            descriptor (ChatProviderDescriptor): Provider static metadata.
            chunks (list[str]): Complete response chunks.
            input_tokens (int): Estimated prompt tokens.
            attempts (int): Attempts made before success.
            provider_index (int): Successful provider index in the chain.
            timeout_seconds (float): Original total deadline.

        Returns:
            ChatCallMetadata: Structured usage, retry, fallback, and cost data.
        """
        usage = _usage_from_provider(provider)
        resolved_input_tokens = usage.input_tokens if usage and usage.input_tokens else input_tokens
        estimated_output_tokens = token_count("".join(chunks))
        output_tokens = (
            usage.output_tokens
            if usage and usage.output_tokens is not None
            else estimated_output_tokens
        )
        return ChatCallMetadata(
            provider_name=descriptor.provider_name,
            model_name=descriptor.model_name,
            configured_provider_name=self.configured_provider_name,
            input_tokens=resolved_input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimate_cost(
                input_tokens=resolved_input_tokens,
                output_tokens=output_tokens,
                input_cost_per_1k=descriptor.input_cost_per_1k,
                output_cost_per_1k=descriptor.output_cost_per_1k,
            ),
            attempts=attempts,
            retry_count=max(attempts - 1, 0),
            fallback_used=provider_index > 0,
            timeout_seconds=timeout_seconds,
        )


class ConfiguredChatProvider:
    """Chat provider resolved from config with local fallback for outages."""

    def __init__(
        self,
        provider_name: str | None,
        *,
        timeout_seconds: float,
        retries: int,
        model_role: str,
        input_cost_per_1k: float,
        output_cost_per_1k: float,
    ) -> None:
        """Create a configured chat provider chain.

        Args:
            provider_name (str | None): Provider name or alias; `llm` uses `LLM_PROVIDER`.
            timeout_seconds (float): Default total deadline for calls.
            retries (int): Retry count per provider.
            model_role (str): Role label, usually `cheap` or `strong`.
            input_cost_per_1k (float): Prompt-token cost estimate for this role.
            output_cost_per_1k (float): Completion-token cost estimate for this role.

        Returns:
            None: A primary provider and optional local fallback are built.
        """
        self.requested_provider_name = provider_name or "llm"
        self.provider_name = resolve_chat_provider_name(self.requested_provider_name)
        self.model_role = model_role
        self.input_cost_per_1k = input_cost_per_1k
        self.output_cost_per_1k = output_cost_per_1k
        primary = self._build_provider()
        providers = [primary]
        if describe_chat_provider(primary).provider_name != "local":
            providers.append(LocalChatProvider(model_name=f"local-{model_role}-fallback"))
        self._provider = FallbackChatProvider(
            providers,
            timeout_seconds=timeout_seconds,
            retries=retries,
            configured_provider_name=self.provider_name,
        )
        self.descriptor = describe_chat_provider(primary)
        self.last_metadata: ChatCallMetadata | None = None

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """Stream through the configured provider chain.

        Args:
            messages (list[ChatMessage]): Guarded prompt messages.
            timeout_seconds (float): Total deadline across attempts.

        Returns:
            AsyncIterator[str]: Complete chunks from the successful provider.
        """
        async for token in self._provider.stream_chat(
            messages,
            timeout_seconds=timeout_seconds,
        ):
            self.last_metadata = self._provider.last_metadata
            yield token
        self.last_metadata = self._provider.last_metadata

    def _build_provider(self) -> ChatProvider:
        """Create the configured primary provider.

        Args:
            None: Provider configuration is read from this instance and settings.

        Returns:
            ChatProvider: Primary provider used before fallback.
        """
        if self.provider_name == "local":
            return LocalChatProvider(model_name=f"local-{self.model_role}")
        if self.provider_name == "openai":
            return OpenAIChatProvider(
                model_name=settings.llm_model,
                api_key=settings.llm_api_key,
                input_cost_per_1k=self.input_cost_per_1k,
                output_cost_per_1k=self.output_cost_per_1k,
            )
        raise ModelProviderConfigError(f"Unsupported chat provider: {self.provider_name}")


@dataclass(frozen=True, slots=True)
class ChatModelCascade:
    """Cheap and strong chat providers used by router and composer."""
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
            model_role="cheap",
            input_cost_per_1k=settings.cheap_model_input_cost_per_1k,
            output_cost_per_1k=settings.cheap_model_output_cost_per_1k,
        ),
        strong=ConfiguredChatProvider(
            settings.strong_chat_provider,
            timeout_seconds=timeout,
            retries=retries,
            model_role="strong",
            input_cost_per_1k=settings.strong_model_input_cost_per_1k,
            output_cost_per_1k=settings.strong_model_output_cost_per_1k,
        ),
    )


def resolve_chat_provider_name(provider_name: str | None) -> str:
    """Resolve chat provider aliases.

    Args:
        provider_name (str | None): Provider value from settings or tests.

    Returns:
        str: Concrete provider name; `llm`, `default`, and `configured` use `LLM_PROVIDER`.
    """
    normalized = _normalize_provider_name(provider_name or "llm")
    if normalized in {"llm", "default", "configured"}:
        return _normalize_provider_name(settings.llm_provider)
    return normalized


def describe_chat_provider(provider: ChatProvider) -> ChatProviderDescriptor:
    """Return metadata for a chat provider.

    Args:
        provider (ChatProvider): Provider instance to inspect.

    Returns:
        ChatProviderDescriptor: Provider descriptor or a conservative fallback descriptor.
    """
    descriptor = getattr(provider, "descriptor", None)
    if isinstance(descriptor, ChatProviderDescriptor):
        return descriptor
    provider_name = _normalize_provider_name(
        str(getattr(provider, "provider_name", provider.__class__.__name__))
    )
    return ChatProviderDescriptor(
        provider_name,
        str(getattr(provider, "model_name", provider_name)),
    )


def get_last_chat_metadata(provider: ChatProvider) -> ChatCallMetadata | None:
    """Return metadata for the provider's most recent gateway-managed call.

    Args:
        provider (ChatProvider): Provider previously used for a chat call.

    Returns:
        ChatCallMetadata | None: Metadata when available, otherwise None.
    """
    metadata = getattr(provider, "last_metadata", None)
    return metadata if isinstance(metadata, ChatCallMetadata) else None


def _tokenize(text: str) -> Sequence[str]:
    """Split text into lowercase whitespace tokens.

    Args:
        text (str):
            Text to tokenize or split into local response chunks.

    Returns:
        Sequence[str]:
            Lowercase non-empty tokens extracted from the text.
    """
    return tuple(token for token in text.lower().split() if token)


def _normalize(vector: list[float]) -> list[float]:
    """L2-normalize a numeric vector.

    Args:
        vector (list[float]):
            Numeric vector to normalize.

    Returns:
        list[float]:
            Normalized vector, or the original zero vector.
    """
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0:
        return vector
    return [component / magnitude for component in vector]


def _local_response_from_prompt(prompt: str) -> str:
    """Build deterministic local text from safe item lines.

    Args:
        prompt (str):
            Prompt text containing optional `SAFE_ITEM` lines.

    Returns:
        str:
            Deterministic response naming only `SAFE_ITEM` entries.
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


def _openai_payload_from_sse_line(line: str) -> dict[str, object] | None:
    """Parse one OpenAI SSE line.

    Args:
        line (str): Raw line from the OpenAI streaming response.

    Returns:
        dict[str, object] | None: Parsed JSON payload, or None for non-content lines.
    """
    if not line.startswith("data: "):
        return None
    data = line.removeprefix("data: ").strip()
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ModelProviderTransientError("Malformed OpenAI SSE payload.") from exc
    if not isinstance(payload, dict):
        raise ModelProviderTransientError("Unexpected OpenAI SSE payload type.")
    return payload


def _openai_token_from_payload(payload: dict[str, object]) -> str | None:
    """Extract a text token from an OpenAI stream payload.

    Args:
        payload (dict[str, object]): Parsed OpenAI SSE payload.

    Returns:
        str | None: Text delta when present.
    """
    choices = payload.get("choices") or []
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    delta = choices[0].get("delta") or {}
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    return content if isinstance(content, str) and content else None


def _openai_token_from_sse_line(line: str) -> str | None:
    """Extract a text token from one raw OpenAI SSE line.

    Args:
        line (str): Raw SSE line.

    Returns:
        str | None: Text delta when present.
    """
    payload = _openai_payload_from_sse_line(line)
    return None if payload is None else _openai_token_from_payload(payload)


def _openai_usage_from_payload(payload: dict[str, object]) -> ChatUsage | None:
    """Extract usage counts from an OpenAI stream payload.

    Args:
        payload (dict[str, object]): Parsed OpenAI SSE payload.

    Returns:
        ChatUsage | None: Usage counts when included by OpenAI.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return ChatUsage(
        input_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        output_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
    )


def _openai_status_error(exc: httpx.HTTPStatusError) -> ModelGatewayError:
    """Map OpenAI status errors to retry policy categories.

    Args:
        exc (httpx.HTTPStatusError): HTTP status error from httpx.

    Returns:
        ModelGatewayError: Categorized gateway error.
    """
    status_code = exc.response.status_code
    if status_code in {401, 403}:
        return ModelProviderAuthError("OpenAI rejected the configured credentials.")
    if 400 <= status_code < 500 and status_code not in {408, 409, 425, 429}:
        return ModelProviderConfigError(f"OpenAI request was rejected with {status_code}.")
    return ModelProviderTransientError(f"OpenAI request failed with {status_code}.")


def _usage_from_provider(provider: ChatProvider) -> ChatUsage | None:
    """Read provider usage from adapters that expose it.

    Args:
        provider (ChatProvider): Provider that completed a call.

    Returns:
        ChatUsage | None: Usage when present and typed.
    """
    usage = getattr(provider, "last_usage", None)
    return usage if isinstance(usage, ChatUsage) else None


def _is_retryable_exception(exc: Exception) -> bool:
    """Return whether an exception may be retried or sent to fallback.

    Args:
        exc (Exception): Provider exception.

    Returns:
        bool: False for config/auth failures; true for transient and unknown failures.
    """
    if isinstance(exc, (ModelProviderConfigError, ModelProviderAuthError)):
        return False
    if isinstance(exc, (ModelProviderTransientError, httpx.TransportError, TimeoutError)):
        return True
    return True


def _normalize_provider_name(provider_name: str) -> str:
    """Normalize provider labels for matching and metric labels.

    Args:
        provider_name (str): Raw provider label.

    Returns:
        str: Lowercase normalized provider label.
    """
    return provider_name.strip().lower().replace("-", "_")


def _chunk_text(text: str, chunk_size: int = 18) -> Sequence[str]:
    """Split local response text into deterministic chunks.

    Args:
        text (str):
            Text to tokenize or split into local response chunks.
        chunk_size (int):
            Maximum number of characters per chunk.

    Returns:
        Sequence[str]:
            Ordered chunks that reconstruct the original text.
    """
    return tuple(text[index : index + chunk_size] for index in range(0, len(text), chunk_size))
