"""Unit tests for retrieval fallback when the Qdrant semantic index is unavailable.

The production path should never fail a menu or policy request only because the
vector index is down. These tests force Qdrant errors and verify that retrieval
continues through the SQL-vector fallback path while recording observability
counters.
"""

from __future__ import annotations

import pytest

from cafe_assistant.observability.metrics import get_metrics_registry
from cafe_assistant.retrieval import policy, vector_store
from cafe_assistant.retrieval.qdrant_store import QdrantVectorStoreError
from cafe_assistant.retrieval.types import SearchHit


class FakePostgresSession:
    """Minimal AsyncSession double exposing a PostgreSQL dialect name."""

    def get_bind(self) -> object:
        """Return an object shaped like SQLAlchemy's bind/dialect pair.

        Args:
            None.

        Returns:
            object:
                Bind-like object with `dialect.name == "postgresql"`.
        """
        dialect = type("Dialect", (), {"name": "postgresql"})()
        return type("Bind", (), {"dialect": dialect})()


async def test_menu_semantic_search_falls_back_to_pgvector_after_qdrant_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify menu semantic retrieval continues after Qdrant raises an adapter error.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Pytest monkeypatch helper used to force Qdrant failure and SQL fallback.

    Returns:
        None:
            Assertions fail if retrieval returns empty or skips fallback metrics.
    """
    registry = get_metrics_registry()
    registry.reset()

    async def broken_qdrant(
        tenant_id: int,
        query_embedding: list[float],
        k: int,
    ) -> list[SearchHit]:
        """Simulate a Qdrant adapter failure for menu vectors.

        Args:
            tenant_id (int):
                Tenant ID passed through from semantic retrieval.
            query_embedding (list[float]):
                Query embedding passed through from semantic retrieval.
            k (int):
                Requested result count passed through from semantic retrieval.

        Returns:
            list[SearchHit]:
                This function never returns because it raises the simulated failure.
        """
        raise QdrantVectorStoreError("qdrant down")

    async def pgvector_catalog_hits(
        session: object,
        tenant_id: int,
        query_embedding: list[float],
        k: int,
    ) -> list[SearchHit]:
        """Return deterministic catalog hits from the SQL-vector fallback path.

        Args:
            session (object):
                Session-like object passed to the fallback function.
            tenant_id (int):
                Tenant ID used by the fallback query.
            query_embedding (list[float]):
                Query embedding used by the fallback query.
            k (int):
                Requested result count for fallback retrieval.

        Returns:
            list[SearchHit]:
                One catalog semantic hit standing in for pgvector results.
        """
        return [SearchHit(item_id=9, score=0.8, source="semantic", rank=1, kind="catalog")]

    monkeypatch.setattr(vector_store, "qdrant_enabled", lambda: True)
    monkeypatch.setattr(vector_store, "search_catalog_item_vectors", broken_qdrant)
    monkeypatch.setattr(vector_store, "_catalog_semantic_search_postgres", pgvector_catalog_hits)

    hits = await vector_store.semantic_search(FakePostgresSession(), 11, [0.1, 0.2], 5)

    assert hits == [SearchHit(item_id=9, score=0.8, source="semantic", rank=1, kind="catalog")]
    snapshot = registry.snapshot()["reliability"]
    assert snapshot["retrieval_qdrant_failures_total{source_kind=catalog_item}"] == 1
    assert (
        snapshot[
            "retrieval_semantic_fallback_total{fallback=pgvector,source_kind=catalog_item}"
        ]
        == 1
    )


async def test_policy_semantic_search_falls_back_to_pgvector_after_qdrant_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify policy semantic retrieval continues after Qdrant raises an adapter error.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Pytest monkeypatch helper used to force Qdrant failure and SQL fallback.

    Returns:
        None:
            Assertions fail if policy retrieval returns empty or skips fallback metrics.
    """
    registry = get_metrics_registry()
    registry.reset()

    async def broken_qdrant(
        tenant_id: int,
        query_embedding: list[float],
        k: int,
    ) -> list[tuple[int, float]]:
        """Simulate a Qdrant adapter failure for policy chunk vectors.

        Args:
            tenant_id (int):
                Tenant ID passed through from semantic retrieval.
            query_embedding (list[float]):
                Query embedding passed through from semantic retrieval.
            k (int):
                Requested result count passed through from semantic retrieval.

        Returns:
            list[tuple[int, float]]:
                This function never returns because it raises the simulated failure.
        """
        raise QdrantVectorStoreError("qdrant down")

    async def pgvector_policy_hits(
        session: object,
        tenant_id: int,
        query_embedding: list[float],
        k: int,
    ) -> list[tuple[int, float]]:
        """Return deterministic policy hits from the SQL-vector fallback path.

        Args:
            session (object):
                Session-like object passed to the fallback function.
            tenant_id (int):
                Tenant ID used by the fallback query.
            query_embedding (list[float]):
                Query embedding used by the fallback query.
            k (int):
                Requested result count for fallback retrieval.

        Returns:
            list[tuple[int, float]]:
                One policy chunk ID and similarity score.
        """
        return [(42, 0.7)]

    monkeypatch.setattr(policy, "qdrant_enabled", lambda: True)
    monkeypatch.setattr(policy, "search_policy_chunk_vectors", broken_qdrant)
    monkeypatch.setattr(policy, "_policy_semantic_search_postgres", pgvector_policy_hits)

    hits = await policy._policy_semantic_search(FakePostgresSession(), 11, [0.1, 0.2], 5)

    assert hits == [(42, 0.7)]
    snapshot = registry.snapshot()["reliability"]
    assert snapshot["retrieval_qdrant_failures_total{source_kind=policy_chunk}"] == 1
    assert (
        snapshot[
            "retrieval_semantic_fallback_total{fallback=pgvector,source_kind=policy_chunk}"
        ]
        == 1
    )