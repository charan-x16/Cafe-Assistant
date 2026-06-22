"""Shared fixtures and helpers for offline agent evaluation.

The evaluation suite must run without network access, live model calls, or a
remote vector service. This module builds isolated in-memory databases, seeds
both the legacy safety fixture and the imported BTB catalog, runs the real chat
agent with deterministic fake providers, and returns structured results that the
individual eval families can score.
"""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cafe_assistant.agent.state_machine import (
    AgentConfig,
    ChatAgent,
    ChatAgentRequest,
    ChatAgentResult,
)
from cafe_assistant.db.base import Base
from cafe_assistant.db.models import MenuItem, Tenant
from cafe_assistant.db.repositories.menu_repo import load_published_catalog_item_views_for_tenant
from cafe_assistant.gateway.model_gateway import ChatMessage, ChatModelCascade
from cafe_assistant.ingestion.btb_markdown import import_btb_documents
from cafe_assistant.memory.session import InMemorySessionMemory
from scripts.embed_catalog import backfill_catalog_embeddings
from tests.fixtures.legacy_embeddings import backfill_menu_embeddings
from tests.fixtures.legacy_menu import TENANT_NAME, seed_database

LEGACY_DATASET_PATH = Path(__file__).parent / "datasets" / "agent_eval_cases.json"
BTB_DATASET_PATH = Path(__file__).parent / "datasets" / "btb_agent_eval_cases.json"


class FakeEmbeddingProvider:
    """Deterministic embedding provider used by all offline eval runs.

    The provider exposes the same simple `embed` interface as production
    providers, but it maps terms into hand-authored semantic buckets. The buckets
    cover both the legacy fixture menu and the imported BTB catalog so exact,
    fuzzy, allergen, policy, and low-sugar style queries can be evaluated without
    downloading a model or calling an external embedding API.
    """

    dimensions = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic vectors for the supplied texts.

        Args:
            texts (list[str]):
                Natural-language queries, menu item documents, or policy chunks
                that need embeddings during an eval run.

        Returns:
            list[list[float]]:
                One 384-dimensional vector per input text. The output order
                matches `texts`, and repeated calls with the same text produce
                byte-for-byte equivalent floating-point values.
        """

        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Convert one text into a normalized feature vector.

        Args:
            text (str):
                Query or catalog text to tokenize. Non-alphanumeric separators
                are ignored, and matching is case-insensitive.

        Returns:
            list[float]:
                A padded 384-dimensional vector. Non-zero vectors are normalized
                so cosine similarity behaves consistently across short and long
                pieces of text.
        """

        tokens = set(re.findall(r"[a-z0-9]+", _normalize_for_match(text)))
        vector = [
            self._feature(
                tokens,
                {"coffee", "espresso", "cappuccino", "latte", "mocha", "americano"},
            ),
            self._feature(tokens, {"tea", "chai", "matcha", "earl", "herbal", "iced"}),
            self._feature(tokens, {"almond", "hazelnut", "nut", "nuts", "pesto", "almnd"}),
            self._feature(tokens, {"cookie", "peanut", "butter"}),
            self._feature(tokens, {"sandwich", "panini", "toast", "meltz", "grilled"}),
            self._feature(tokens, {"chocolate", "mocha", "brownie", "shake"}),
            self._feature(tokens, {"gluten", "bread", "sourdough", "dough", "garlic", "jalapeno"}),
            self._feature(tokens, {"vegan", "vegetarian", "gluten_free", "glutenfree"}),
            self._feature(tokens, {"pizza", "pesto", "margherita", "paneer"}),
            self._feature(tokens, {"chicken", "wings", "bbq", "barbeque"}),
            self._feature(tokens, {"policy", "refund", "payment", "cancellation", "replacement"}),
            self._feature(tokens, {"cold", "iced", "mocktail", "shake", "smoothie"}),
        ]
        magnitude = math.sqrt(sum(component * component for component in vector))
        if magnitude == 0:
            return self._pad(vector)
        return self._pad([component / magnitude for component in vector])

    def _feature(self, tokens: set[str], vocabulary: set[str]) -> float:
        """Count how strongly a token set matches one semantic bucket.

        Args:
            tokens (set[str]):
                Normalized tokens extracted from the text being embedded.
            vocabulary (set[str]):
                Terms that define one semantic feature in the fake embedding
                space.

        Returns:
            float:
                The number of overlapping terms, represented as a float so the
                value can be normalized with the rest of the vector.
        """

        return float(len(tokens & vocabulary))

    def _pad(self, vector: list[float]) -> list[float]:
        """Pad a low-dimensional semantic vector to the configured size.

        Args:
            vector (list[float]):
                The compact feature vector produced by `_embed_one` before it is
                shaped like the real embedding model output.

        Returns:
            list[float]:
                A vector with exactly `dimensions` entries. Existing feature
                values are preserved and trailing dimensions are zeros.
        """

        return vector + [0.0] * (self.dimensions - len(vector))


class CapturingChatProvider:
    """Fake streaming chat provider that records prompt context.

    The provider does not reason about safety. It simply echoes the `SAFE_ITEM`
    names that the composer placed in the model context, which lets evals verify
    that unsafe menu items never reached generation and never appear in the final
    customer-facing response.
    """

    def __init__(self) -> None:
        """Initialize an empty capture buffer for model calls.

        Args:
            None.

        Returns:
            None:
                The provider stores future calls in `self.calls` for eval
                assertions; no value is returned.
        """

        self.calls: list[list[ChatMessage]] = []

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """Stream a deterministic response based only on safe item lines.

        Args:
            messages (list[ChatMessage]):
                Prompt messages created by the composer. The fake provider reads
                only lines beginning with `SAFE_ITEM:`.
            timeout_seconds (float):
                The timeout configured by the caller. It is accepted to match the
                production protocol but not used by this deterministic provider.

        Returns:
            AsyncIterator[str]:
                Chunks of one synthetic response. If no safe items are present,
                the response says no menu item can be suggested from the safe set.
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
    """One labeled adversarial or quality case for the full chat agent.

    Args:
        id (str): Stable identifier used in reports and traces.
        query (str): Customer message sent to the agent.
        unsafe_item_names (list[str]): Menu item names that must never be
            recommended or mentioned as recommendations for this case.
        expected_any_item_names (list[str]): Acceptable relevant item names; at
            least one must appear when the list is non-empty.
        expect_empty_safe_set (bool): Whether the agent should end with no safe
            recommendations for this case.
        expect_medical_refusal (bool): Whether the response must include the
            medical disclaimer/refusal path and return no items.
        source (str): Dataset family that produced the case.
    """

    id: str
    query: str
    unsafe_item_names: list[str]
    expected_any_item_names: list[str]
    expect_empty_safe_set: bool
    expect_medical_refusal: bool
    source: str = "legacy_seed"


@dataclass(slots=True)
class EvalRunResult:
    """Observed output from running one eval case through the chat agent.

    Args:
        case (EvalCase): The labeled input case.
        result (ChatAgentResult): Agent output, including final response and safe
            item list.
        latency_ms (float): End-to-end agent latency in milliseconds.
        menu_names (set[str]): Valid menu item names for the tenant under test.
        model_messages (list[ChatMessage]): Strong-model context captured for
            safety and groundedness inspection.
    """

    case: EvalCase
    result: ChatAgentResult
    latency_ms: float
    menu_names: set[str]
    model_messages: list[ChatMessage]

    @property
    def recommended_names(self) -> set[str]:
        """Return the structured item names recommended by the agent.

        Args:
            None.

        Returns:
            set[str]:
                Names from `result.safe_items`. These are treated as explicit
                recommendations and are scored by allergen and relevance evals.
        """

        return {item.name for item in self.result.safe_items}


async def run_eval_cases(*, include_catalog: bool = True) -> list[EvalRunResult]:
    """Run all configured full-agent eval datasets.

    Args:
        include_catalog (bool):
            When true, run both the legacy fixture cases and the imported BTB
            catalog cases. When false, run only the small legacy fixture for fast
            local debugging.

    Returns:
        list[EvalRunResult]:
            One structured result per dataset case, preserving dataset order with
            legacy cases first and BTB catalog cases second.
    """

    results = await _run_legacy_eval_cases()
    if include_catalog:
        results.extend(await _run_btb_catalog_eval_cases())
    return results


async def _run_legacy_eval_cases() -> list[EvalRunResult]:
    """Run the historical seed-menu eval cases in an isolated database.

    Args:
        None.

    Returns:
        list[EvalRunResult]:
            Results for the legacy fixture menu. This dataset remains useful
            because it has compact, hand-labeled cases for incomplete allergen
            data, diabetes wording, and prompt-injection attempts.
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
            agent = _build_agent(session, memory, cheap_model, strong_model, provider)
            return [
                await _run_one_case(
                    agent=agent,
                    case=case,
                    tenant_id=tenant_id,
                    menu_names=menu_names,
                    strong_model=strong_model,
                )
                for case in load_cases(LEGACY_DATASET_PATH, source="legacy_seed")
            ]
    finally:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await engine.dispose()


async def _run_btb_catalog_eval_cases() -> list[EvalRunResult]:
    """Run full-agent eval cases against the imported BTB catalog.

    Args:
        None.

    Returns:
        list[EvalRunResult]:
            Results for the real markdown-derived catalog. These cases protect
            the production ingestion path, catalog retrieval, and item-name
            parsing from drifting away from the safety gate.
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
            import_result = await import_btb_documents(session)
            await backfill_catalog_embeddings(
                session,
                provider=provider,
                tenant_id=import_result.tenant_id,
            )
            catalog_views = await load_published_catalog_item_views_for_tenant(
                session,
                import_result.tenant_id,
            )
            menu_names = {view.name for view in catalog_views}
            agent = _build_agent(session, memory, cheap_model, strong_model, provider)
            return [
                await _run_one_case(
                    agent=agent,
                    case=case,
                    tenant_id=import_result.tenant_id,
                    menu_names=menu_names,
                    strong_model=strong_model,
                )
                for case in load_cases(BTB_DATASET_PATH, source="btb_catalog")
            ]
    finally:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await engine.dispose()


def _build_agent(
    session: AsyncSession,
    memory: InMemorySessionMemory,
    cheap_model: CapturingChatProvider,
    strong_model: CapturingChatProvider,
    provider: FakeEmbeddingProvider,
) -> ChatAgent:
    """Create a chat agent configured for deterministic eval execution.

    Args:
        session (AsyncSession): Database session containing the dataset tenant.
        memory (InMemorySessionMemory): Isolated session-memory backend for the
            eval run.
        cheap_model (CapturingChatProvider): Fake provider used by routing and
            low-confidence classification paths.
        strong_model (CapturingChatProvider): Fake provider used for final
            response composition.
        provider (FakeEmbeddingProvider): Deterministic embedding provider used
            by retrieval.

    Returns:
        ChatAgent:
            A real chat-agent instance with mocked model/embedding dependencies
            and production orchestration limits suitable for offline tests.
    """

    return ChatAgent(
        session,
        memory=memory,
        chat_models=ChatModelCascade(cheap=cheap_model, strong=strong_model),
        embedding_provider=provider,
        config=AgentConfig(max_tool_calls=4, deadline_seconds=10.0, search_k=8),
    )


async def _run_one_case(
    *,
    agent: ChatAgent,
    case: EvalCase,
    tenant_id: int,
    menu_names: set[str],
    strong_model: CapturingChatProvider,
) -> EvalRunResult:
    """Execute one labeled case through the real chat-agent state machine.

    Args:
        agent (ChatAgent): Configured agent instance for the dataset tenant.
        case (EvalCase): Labeled query and expected safety outcomes.
        tenant_id (int): Tenant identifier scoped to the dataset database.
        menu_names (set[str]): Valid menu names used for response parsing.
        strong_model (CapturingChatProvider): Fake strong model whose latest
            prompt context should be captured for this case.

    Returns:
        EvalRunResult:
            The agent output, latency, menu vocabulary, and per-case model
            messages. If the state machine used a deterministic fallback and did
            not call the strong model, `model_messages` is empty for the case.
    """

    call_count_before = len(strong_model.calls)
    started_at = time.perf_counter()
    result = await agent.run(
        ChatAgentRequest(
            session_id=f"eval-{case.source}-{case.id}",
            tenant_id=tenant_id,
            message=case.query,
            request_id=f"eval-{case.source}-{case.id}",
            trace_id=f"eval-{case.source}-{case.id}",
        )
    )
    latency_ms = (time.perf_counter() - started_at) * 1000.0
    model_messages = (
        list(strong_model.calls[-1]) if len(strong_model.calls) > call_count_before else []
    )
    return EvalRunResult(
        case=case,
        result=result,
        latency_ms=latency_ms,
        menu_names=menu_names,
        model_messages=model_messages,
    )


def load_cases(path: Path = LEGACY_DATASET_PATH, *, source: str = "legacy_seed") -> list[EvalCase]:
    """Load labeled eval cases from a JSON dataset file.

    Args:
        path (Path): JSON file containing a list of eval case objects.
        source (str): Dataset label attached to each returned case so reports can
            distinguish legacy fixture failures from BTB catalog failures.

    Returns:
        list[EvalCase]:
            Parsed eval cases with defensive type conversion for strings, lists,
            and booleans used by the scoring functions.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=str(item["id"]),
            query=str(item["query"]),
            unsafe_item_names=list(item["unsafe_item_names"]),
            expected_any_item_names=list(item["expected_any_item_names"]),
            expect_empty_safe_set=bool(item["expect_empty_safe_set"]),
            expect_medical_refusal=bool(item["expect_medical_refusal"]),
            source=source,
        )
        for item in payload
    ]


def parse_response_item_names(response: str, menu_names: set[str]) -> set[str]:
    """Find known menu item names mentioned in a generated response.

    Args:
        response (str): Customer-facing assistant text to inspect.
        menu_names (set[str]): Authoritative menu item names for the current
            tenant. Only these names can be returned by the parser.

    Returns:
        set[str]:
            Menu names whose normalized form appears in the normalized response.
            Matching ignores case, punctuation, accents, and repeated whitespace
            so the safety gate still detects unsafe mentions in slightly varied
            model output.
    """

    normalized_response = f" {_normalize_for_match(response)} "
    matched: set[str] = set()
    for name in menu_names:
        normalized_name = _normalize_for_match(name)
        if normalized_name and f" {normalized_name} " in normalized_response:
            matched.add(name)
    return matched


def _normalize_for_match(text: str) -> str:
    """Normalize human text for deterministic item-name matching.

    Args:
        text (str): Raw model response, query, or menu name.

    Returns:
        str:
            Lowercase ASCII-like text with accents stripped, punctuation replaced
            by spaces, and whitespace collapsed to single separators.
    """

    decomposed = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    normalized = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return re.sub(r"\s+", " ", normalized).strip()


def _chunks(text: str, chunk_size: int = 12) -> list[str]:
    """Split a fake model response into streaming-sized chunks.

    Args:
        text (str): Full synthetic response to stream.
        chunk_size (int): Maximum number of characters per emitted chunk.

    Returns:
        list[str]:
            Ordered response chunks. Concatenating the chunks recreates `text`.
    """

    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
