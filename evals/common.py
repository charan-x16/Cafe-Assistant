"""Evaluation helpers for deterministic quality and safety gates.
Supports offline eval execution without external model calls.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cafe_assistant.agent.state_machine import (
    AgentConfig,
    ChatAgent,
    ChatAgentRequest,
    ChatAgentResult,
)
from cafe_assistant.db.base import Base
from cafe_assistant.db.models import MenuItem, Tenant
from cafe_assistant.gateway.model_gateway import ChatMessage, ChatModelCascade
from cafe_assistant.memory.session import InMemorySessionMemory
from tests.fixtures.legacy_embeddings import backfill_menu_embeddings
from tests.fixtures.legacy_menu import TENANT_NAME, seed_database

DATASET_PATH = Path(__file__).parent / "datasets" / "agent_eval_cases.json"


class FakeEmbeddingProvider:
    """Container for fake embedding provider behavior and data."""
    dimensions = 384

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
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        vector = [
            self._feature(tokens, {"coffee", "espresso", "cappuccino", "latte", "mocha"}),
            self._feature(tokens, {"tea", "chai", "matcha", "earl"}),
            self._feature(tokens, {"almond", "almnd", "nut"}),
            self._feature(tokens, {"cookie", "peanut", "butter"}),
            self._feature(tokens, {"sandwich", "panini", "toast"}),
            self._feature(tokens, {"chocolate", "mocha"}),
            self._feature(tokens, {"gluten", "bread", "sourdough", "dough"}),
            self._feature(tokens, {"vegan", "vegetarian", "gluten_free"}),
        ]
        magnitude = math.sqrt(sum(component * component for component in vector))
        if magnitude == 0:
            return self._pad(vector)
        return self._pad([component / magnitude for component in vector])

    def _feature(self, tokens: set[str], vocabulary: set[str]) -> float:
        """Handle feature.

        Args:
            tokens (set[str]):
                Tokens value required to perform this operation.
            vocabulary (set[str]):
                Vocabulary value required to perform this operation.

        Returns:
            float:
                Value produced for the caller according to the function contract.
        """
        return float(len(tokens & vocabulary))

    def _pad(self, vector: list[float]) -> list[float]:
        """Handle pad.

        Args:
            vector (list[float]):
                Vector being normalized, converted, or sent to the vector store.

        Returns:
            list[float]:
                Value produced for the caller according to the function contract.
        """
        return vector + [0.0] * (self.dimensions - len(vector))


class CapturingChatProvider:
    """Container for capturing chat provider behavior and data."""
    def __init__(self) -> None:
        """Initialize the object with the dependencies or values required later.

        Args:
            None.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
        self.calls: list[list[ChatMessage]] = []

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
        self.calls.append(messages)
        item_names = [
            line.removeprefix("SAFE_ITEM:").split("|", 1)[0].strip()
            for message in messages
            for line in message.content.splitlines()
            if line.startswith("SAFE_ITEM:")
        ]
        response = (
            "I can suggest " + ", ".join(item_names) + "."
            if item_names
            else "I cannot suggest a menu item from the provided safe set."
        )
        for chunk in _chunks(response):
            yield chunk


@dataclass(frozen=True, slots=True)
class EvalCase:
    """Container for eval case behavior and data."""
    id: str
    query: str
    unsafe_item_names: list[str]
    expected_any_item_names: list[str]
    expect_empty_safe_set: bool
    expect_medical_refusal: bool


@dataclass(slots=True)
class EvalRunResult:
    """Container for eval run result behavior and data."""
    case: EvalCase
    result: ChatAgentResult
    latency_ms: float
    menu_names: set[str]
    model_messages: list[ChatMessage]

    @property
    def recommended_names(self) -> set[str]:
        """Handle recommended names.

        Args:
            None.

        Returns:
            set[str]:
                Value produced for the caller according to the function contract.
        """
        return {item.name for item in self.result.safe_items}


async def run_eval_cases() -> list[EvalRunResult]:
    """Run eval cases.

    Args:
        None.

    Returns:
        list[EvalRunResult]:
            Value produced for the caller according to the function contract.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    try:
        async with session_factory() as session:
            provider = FakeEmbeddingProvider()
            strong_model = CapturingChatProvider()
            cheap_model = CapturingChatProvider()
            memory = InMemorySessionMemory()
            await seed_database(session)
            tenant_id = await session.scalar(select(Tenant.id).where(Tenant.name == TENANT_NAME))
            if tenant_id is None:
                raise RuntimeError("Seed tenant not found.")
            await backfill_menu_embeddings(session, provider=provider, tenant_id=tenant_id)
            menu_names = set(await session.scalars(select(MenuItem.name)))
            agent = ChatAgent(
                session,
                memory=memory,
                chat_models=ChatModelCascade(cheap=cheap_model, strong=strong_model),
                embedding_provider=provider,
                config=AgentConfig(max_tool_calls=4, deadline_seconds=10.0, search_k=8),
            )
            results: list[EvalRunResult] = []
            for case in load_cases():
                started_at = time.perf_counter()
                result = await agent.run(
                    ChatAgentRequest(
                        session_id=f"eval-{case.id}",
                        tenant_id=tenant_id,
                        message=case.query,
                        request_id=f"eval-{case.id}",
                        trace_id=f"eval-{case.id}",
                    )
                )
                latency_ms = (time.perf_counter() - started_at) * 1000.0
                results.append(
                    EvalRunResult(
                        case=case,
                        result=result,
                        latency_ms=latency_ms,
                        menu_names=menu_names,
                        model_messages=list(strong_model.calls[-1]) if strong_model.calls else [],
                    )
                )
            return results
    finally:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await engine.dispose()


def load_cases() -> list[EvalCase]:
    """Load cases.

    Args:
        None.

    Returns:
        list[EvalCase]:
            Loaded records or projected domain values matching the requested scope.
    """
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=str(item["id"]),
            query=str(item["query"]),
            unsafe_item_names=list(item["unsafe_item_names"]),
            expected_any_item_names=list(item["expected_any_item_names"]),
            expect_empty_safe_set=bool(item["expect_empty_safe_set"]),
            expect_medical_refusal=bool(item["expect_medical_refusal"]),
        )
        for item in payload
    ]


def parse_response_item_names(response: str, menu_names: set[str]) -> set[str]:
    """Parse response item names.

    Args:
        response (str):
            HTTP or chat response object being parsed or checked.
        menu_names (set[str]):
            Menu names value required to perform this operation.

    Returns:
        set[str]:
            Parsed values extracted from the source text or structured payload.
    """
    return {name for name in menu_names if name in response}


def _chunks(text: str, chunk_size: int = 12) -> list[str]:
    """Handle chunks.

    Args:
        text (str):
            Input text to normalize, embed, tokenize, or classify.
        chunk_size (int):
            Chunk size value required to perform this operation.

    Returns:
        list[str]:
            Value produced for the caller according to the function contract.
    """
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
