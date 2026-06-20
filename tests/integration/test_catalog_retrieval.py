"""Tests for catalog retrieval.
Exercises expected behavior with deterministic fixtures and mocked providers where needed.
"""

from __future__ import annotations

import math
import re
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cafe_assistant.db.base import Base
from cafe_assistant.db.models import CatalogItemEmbedding, PolicyChunkEmbedding
from cafe_assistant.domain.dietary import AllergenCode, CustomerRestrictions
from cafe_assistant.gateway.model_gateway import EmbeddingProvider
from cafe_assistant.ingestion.btb_markdown import import_btb_documents
from cafe_assistant.retrieval.hybrid import hybrid_search, search_menu
from cafe_assistant.retrieval.policy import search_policy_chunks
from scripts.embed_catalog import backfill_catalog_embeddings


class FakeCatalogEmbeddingProvider:
    """Container for fake catalog embedding provider behavior and data."""
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
            self._feature(tokens, {"pizza", "pesto", "margherita", "paneer"}),
            self._feature(tokens, {"garlic", "bread", "jalape", "jalapeno"}),
            self._feature(tokens, {"chicken", "wings", "bbq", "barbeque"}),
            self._feature(tokens, {"vegan", "vegetarian", "gluten", "gluten_free"}),
            self._feature(tokens, {"almond", "hazelnut", "nut", "nuts", "pesto"}),
            self._feature(tokens, {"policy", "refund", "payment", "cancellation"}),
            self._feature(tokens, {"cold", "iced", "mocktail", "shake"}),
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


@pytest.fixture
async def catalog_retrieval_session() -> AsyncIterator[
    tuple[AsyncSession, int, EmbeddingProvider]
]:
    """Handle catalog retrieval session.

    Args:
        None.

    Returns:
        AsyncIterator[tuple[AsyncSession, int, EmbeddingProvider]]:
            Streamed values yielded to the caller as they become available.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        provider = FakeCatalogEmbeddingProvider()
        result = await import_btb_documents(session)
        await backfill_catalog_embeddings(
            session,
            provider=provider,
            tenant_id=result.tenant_id,
        )
        yield session, result.tenant_id, provider

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def test_catalog_embedding_backfill_includes_items_and_policy_chunks(
    catalog_retrieval_session: tuple[AsyncSession, int, EmbeddingProvider],
) -> None:
    """Verify that catalog embedding backfill includes items and policy chunks.

    Args:
        catalog_retrieval_session (tuple[AsyncSession, int, EmbeddingProvider]):
            Catalog retrieval session value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    session, _tenant_id, _provider = catalog_retrieval_session

    item_embedding_count = await session.scalar(
        select(func.count()).select_from(CatalogItemEmbedding)
    )
    policy_embedding_count = await session.scalar(
        select(func.count()).select_from(PolicyChunkEmbedding)
    )

    assert item_embedding_count is not None
    assert item_embedding_count >= 100
    assert policy_embedding_count is not None
    assert policy_embedding_count >= 25


async def test_hybrid_search_returns_catalog_hits(
    catalog_retrieval_session: tuple[AsyncSession, int, EmbeddingProvider],
) -> None:
    """Verify that hybrid search returns catalog hits.

    Args:
        catalog_retrieval_session (tuple[AsyncSession, int, EmbeddingProvider]):
            Catalog retrieval session value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    session, tenant_id, provider = catalog_retrieval_session

    hits = await hybrid_search(
        session,
        tenant_id,
        "cappuccino",
        k=5,
        embedding_provider=provider,
    )

    assert hits
    assert {hit.kind for hit in hits} == {"catalog"}


async def test_exact_catalog_query_returns_matching_item(
    catalog_retrieval_session: tuple[AsyncSession, int, EmbeddingProvider],
) -> None:
    """Verify that exact catalog query returns matching item.

    Args:
        catalog_retrieval_session (tuple[AsyncSession, int, EmbeddingProvider]):
            Catalog retrieval session value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    session, tenant_id, provider = catalog_retrieval_session

    results = await search_menu(
        session,
        tenant_id,
        "cappuccino",
        CustomerRestrictions(avoid_allergens=set(), modes=set()),
        embedding_provider=provider,
    )

    assert results
    assert results[0].name.startswith("Cappuccino")


async def test_fuzzy_catalog_query_returns_plausible_item(
    catalog_retrieval_session: tuple[AsyncSession, int, EmbeddingProvider],
) -> None:
    """Verify that fuzzy catalog query returns plausible item.

    Args:
        catalog_retrieval_session (tuple[AsyncSession, int, EmbeddingProvider]):
            Catalog retrieval session value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    session, tenant_id, provider = catalog_retrieval_session

    results = await search_menu(
        session,
        tenant_id,
        "jalapeno garlick bread",
        CustomerRestrictions(avoid_allergens=set(), modes=set()),
        embedding_provider=provider,
    )

    assert "Jalapeño Garlic Bread" in {item.name for item in results[:5]}


async def test_catalog_retrieval_still_filters_unsafe_semantic_match(
    catalog_retrieval_session: tuple[AsyncSession, int, EmbeddingProvider],
) -> None:
    """Verify that catalog retrieval still filters unsafe semantic match.

    Args:
        catalog_retrieval_session (tuple[AsyncSession, int, EmbeddingProvider]):
            Catalog retrieval session value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    session, tenant_id, provider = catalog_retrieval_session

    results = await search_menu(
        session,
        tenant_id,
        "pesto pizza",
        CustomerRestrictions(avoid_allergens={AllergenCode.TREE_NUT}, modes=set()),
        k=10,
        embedding_provider=provider,
    )

    assert "Veg Pesto Pizza" not in {item.name for item in results}
    assert all(AllergenCode.TREE_NUT not in item.allergen_codes for item in results)
    assert all(item.allergen_data_complete for item in results)


async def test_policy_chunk_search_uses_policy_corpus(
    catalog_retrieval_session: tuple[AsyncSession, int, EmbeddingProvider],
) -> None:
    """Verify that policy chunk search uses policy corpus.

    Args:
        catalog_retrieval_session (tuple[AsyncSession, int, EmbeddingProvider]):
            Catalog retrieval session value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    session, tenant_id, provider = catalog_retrieval_session

    results = await search_policy_chunks(
        session,
        tenant_id,
        "refund for wrong item served",
        k=3,
        embedding_provider=provider,
    )

    assert results
    combined_text = " ".join(f"{result.heading_path} {result.content}" for result in results)
    assert "refund" in combined_text.lower() or "replacement" in combined_text.lower()
