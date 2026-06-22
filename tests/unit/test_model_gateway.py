"""Unit tests for the production model gateway behavior.

The tests use deterministic fake providers only. They verify retry/fallback
semantics, safe buffered streaming, timeout handling, provider alias resolution,
OpenAI SSE parsing, and cost metadata without making network calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from cafe_assistant.gateway import model_gateway
from cafe_assistant.gateway.model_gateway import (
    ChatMessage,
    ChatProviderDescriptor,
    ChatUsage,
    ConfiguredChatProvider,
    FallbackChatProvider,
    ModelProviderAuthError,
    ModelProviderTimeoutError,
    ModelProviderTransientError,
    _openai_token_from_sse_line,
    _openai_usage_from_payload,
    resolve_chat_provider_name,
)


class ScriptedChatProvider:
    """Fake provider with scripted chunks, failures, delay, and usage metadata."""

    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        chunks: list[str] | None = None,
        failures: list[Exception] | None = None,
        partial_chunks_before_failure: list[str] | None = None,
        delay_seconds: float = 0.0,
        usage: ChatUsage | None = None,
        input_cost_per_1k: float = 0.0,
        output_cost_per_1k: float = 0.0,
    ) -> None:
        """Create a scripted fake provider.

        Args:
            provider_name (str): Provider label exposed through the descriptor.
            model_name (str): Model label exposed through the descriptor.
            chunks (list[str] | None): Chunks yielded on successful attempts.
            failures (list[Exception] | None): Exceptions raised on early calls.
            partial_chunks_before_failure (list[str] | None): Chunks yielded before a failure.
            delay_seconds (float): Delay before yielding, used for timeout tests.
            usage (ChatUsage | None): Usage exposed after a successful call.
            input_cost_per_1k (float): Prompt-token cost estimate.
            output_cost_per_1k (float): Completion-token cost estimate.

        Returns:
            None: The provider stores the script for future calls.
        """
        self.descriptor = ChatProviderDescriptor(
            provider_name=provider_name,
            model_name=model_name,
            input_cost_per_1k=input_cost_per_1k,
            output_cost_per_1k=output_cost_per_1k,
        )
        self.chunks = chunks or []
        self.failures = list(failures or [])
        self.partial_chunks_before_failure = partial_chunks_before_failure or []
        self.delay_seconds = delay_seconds
        self.last_usage = usage
        self.calls = 0

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """Stream the scripted response or raise the scripted failure.

        Args:
            messages (list[ChatMessage]): Prompt messages supplied by the gateway.
            timeout_seconds (float): Timeout supplied by the gateway.

        Returns:
            AsyncIterator[str]: Scripted text chunks when the call succeeds.
        """
        del messages, timeout_seconds
        self.calls += 1
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.failures:
            failure = self.failures.pop(0)
            for chunk in self.partial_chunks_before_failure:
                yield chunk
            raise failure
        for chunk in self.chunks:
            yield chunk


async def _collect(provider: FallbackChatProvider, *, timeout_seconds: float = 1.0) -> str:
    """Collect a provider stream into one string.

    Args:
        provider (FallbackChatProvider): Gateway provider under test.
        timeout_seconds (float): Total timeout passed to the gateway call.

    Returns:
        str: Concatenated output chunks.
    """
    chunks: list[str] = []
    async for chunk in provider.stream_chat(
        [ChatMessage(role="user", content="hello cafe")],
        timeout_seconds=timeout_seconds,
    ):
        chunks.append(chunk)
    return "".join(chunks)


async def test_retry_succeeds_without_using_fallback() -> None:
    """A retryable first failure should retry the same provider before fallback."""
    primary = ScriptedChatProvider(
        provider_name="primary",
        model_name="primary-model",
        failures=[ModelProviderTransientError("temporary")],
        chunks=["primary ok"],
    )
    fallback = ScriptedChatProvider(
        provider_name="fallback",
        model_name="fallback-model",
        chunks=["fallback"],
    )
    provider = FallbackChatProvider(
        [primary, fallback],
        timeout_seconds=1.0,
        retries=1,
        configured_provider_name="primary",
    )

    output = await _collect(provider)

    assert output == "primary ok"
    assert primary.calls == 2
    assert fallback.calls == 0
    assert provider.last_metadata is not None
    assert provider.last_metadata.retry_count == 1
    assert provider.last_metadata.fallback_used is False


async def test_failed_partial_stream_is_not_mixed_with_fallback() -> None:
    """Chunks from a failed primary provider must not be yielded before fallback."""
    primary = ScriptedChatProvider(
        provider_name="primary",
        model_name="primary-model",
        failures=[ModelProviderTransientError("stream broke")],
        partial_chunks_before_failure=["bad partial"],
    )
    fallback = ScriptedChatProvider(
        provider_name="fallback",
        model_name="fallback-model",
        chunks=["safe fallback"],
    )
    provider = FallbackChatProvider(
        [primary, fallback],
        timeout_seconds=1.0,
        retries=0,
        configured_provider_name="primary",
    )

    output = await _collect(provider)

    assert output == "safe fallback"
    assert "bad partial" not in output
    assert provider.last_metadata is not None
    assert provider.last_metadata.fallback_used is True
    assert provider.last_metadata.provider_name == "fallback"


async def test_auth_error_does_not_retry_or_fallback() -> None:
    """Permanent authentication failures should stop the provider chain."""
    primary = ScriptedChatProvider(
        provider_name="primary",
        model_name="primary-model",
        failures=[ModelProviderAuthError("bad key")],
    )
    fallback = ScriptedChatProvider(
        provider_name="fallback",
        model_name="fallback-model",
        chunks=["fallback"],
    )
    provider = FallbackChatProvider(
        [primary, fallback],
        timeout_seconds=1.0,
        retries=2,
        configured_provider_name="primary",
    )

    with pytest.raises(ModelProviderAuthError):
        await _collect(provider)

    assert primary.calls == 1
    assert fallback.calls == 0


async def test_global_timeout_bounds_retries_and_fallback() -> None:
    """The total timeout should apply to the whole provider chain."""
    slow = ScriptedChatProvider(
        provider_name="slow",
        model_name="slow-model",
        chunks=["late"],
        delay_seconds=0.05,
    )
    fallback = ScriptedChatProvider(
        provider_name="fallback",
        model_name="fallback-model",
        chunks=["fallback"],
    )
    provider = FallbackChatProvider(
        [slow, fallback],
        timeout_seconds=0.01,
        retries=1,
        configured_provider_name="slow",
    )

    with pytest.raises(ModelProviderTimeoutError):
        await _collect(provider, timeout_seconds=0.01)

    assert fallback.calls == 0


def test_provider_alias_resolution_uses_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `llm` alias should resolve through the configured LLM provider."""
    monkeypatch.setattr(model_gateway.settings, "llm_provider", "local")

    assert resolve_chat_provider_name("llm") == "local"
    provider = ConfiguredChatProvider(
        "llm",
        timeout_seconds=1.0,
        retries=0,
        model_role="cheap",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
    )

    assert provider.descriptor.provider_name == "local"


def test_openai_sse_parsing_and_usage_extraction() -> None:
    """OpenAI SSE helpers should parse tokens and optional usage chunks."""
    token_line = 'data: {"choices":[{"delta":{"content":"Hi"}}]}'
    usage_payload = {
        "choices": [],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }

    assert _openai_token_from_sse_line(token_line) == "Hi"
    assert _openai_token_from_sse_line("data: [DONE]") is None
    assert _openai_usage_from_payload(usage_payload) == ChatUsage(
        input_tokens=12,
        output_tokens=4,
    )


async def test_metadata_uses_provider_usage_for_cost() -> None:
    """Gateway metadata should prefer provider usage over local token estimates."""
    provider = ScriptedChatProvider(
        provider_name="metered",
        model_name="metered-model",
        chunks=["hello world"],
        usage=ChatUsage(input_tokens=10, output_tokens=5),
        input_cost_per_1k=0.1,
        output_cost_per_1k=0.2,
    )
    gateway = FallbackChatProvider(
        [provider],
        timeout_seconds=1.0,
        retries=0,
        configured_provider_name="metered",
    )

    output = await _collect(gateway)

    assert output == "hello world"
    assert gateway.last_metadata is not None
    assert gateway.last_metadata.input_tokens == 10
    assert gateway.last_metadata.output_tokens == 5
    assert gateway.last_metadata.estimated_cost_usd == pytest.approx(0.002)
