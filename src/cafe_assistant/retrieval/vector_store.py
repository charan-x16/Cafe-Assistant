"""Implementation module for vector store.
Contains typed helpers used by the cafe assistant backend runtime.
"""

from __future__ import annotations

import math

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.models import (
    CatalogItem,
    CatalogItemEmbedding,
    CatalogItemVariant,
    Menu,
    MenuItem,
    MenuVersion,
)
from cafe_assistant.retrieval.qdrant_store import qdrant_enabled, search_catalog_item_vectors
from cafe_assistant.retrieval.types import SearchHit


async def semantic_search(
    session: AsyncSession,
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[SearchHit]:
    """Search menu records with embedding similarity.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query_embedding (list[float]):
            Embedding vector produced from the user query.
        k (int):
            Maximum number of ranked candidates or results to return.

    Returns:
        list[SearchHit]:
            Ranked semantic hits from Qdrant, pgvector, or local cosine scoring.
    """
    if k <= 0:
        return []

    bind = session.get_bind()
    if bind.dialect.name == "postgresql" and qdrant_enabled():
        try:
            return await search_catalog_item_vectors(tenant_id, query_embedding, k)
        except RuntimeError:
            return []

    if bind.dialect.name == "postgresql":
        catalog_hits = await _catalog_semantic_search_postgres(
            session,
            tenant_id,
            query_embedding,
            k,
        )
        if catalog_hits:
            return catalog_hits
        return await _legacy_semantic_search_postgres(session, tenant_id, query_embedding, k)

    catalog_hits = await _catalog_semantic_search_python(session, tenant_id, query_embedding, k)
    if catalog_hits:
        return catalog_hits
    return await _legacy_semantic_search_python(session, tenant_id, query_embedding, k)


async def _catalog_semantic_search_postgres(
    session: AsyncSession,
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[SearchHit]:
    """Handle catalog semantic search postgres.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query_embedding (list[float]):
            Embedding vector produced from the user query.
        k (int):
            Maximum number of ranked candidates or results to return.

    Returns:
        list[SearchHit]:
            Value produced for the caller according to the function contract.
    """
    vector_literal = _to_pgvector_literal(query_embedding)
    result = await session.execute(
        text(
            """
            SELECT
                catalog_item_variants.id,
                1 - (catalog_item_embeddings.embedding <=> CAST(:query_embedding AS vector))
                    AS score
            FROM catalog_item_embeddings
            JOIN catalog_item_variants
                ON catalog_item_variants.id = catalog_item_embeddings.variant_id
            JOIN catalog_items
                ON catalog_items.id = catalog_item_variants.catalog_item_id
            JOIN menu_versions
                ON menu_versions.id = catalog_items.menu_version_id
            JOIN menus
                ON menus.id = menu_versions.menu_id
            WHERE menus.tenant_id = :tenant_id
              AND menu_versions.status = 'published'
              AND catalog_items.is_available = true
              AND catalog_item_variants.is_available = true
              AND catalog_item_embeddings.embedding IS NOT NULL
            ORDER BY catalog_item_embeddings.embedding <=> CAST(:query_embedding AS vector),
                     catalog_item_variants.id
            LIMIT :limit
            """
        ),
        {
            "tenant_id": tenant_id,
            "query_embedding": vector_literal,
            "limit": k,
        },
    )
    return [
        SearchHit(
            item_id=row.id,
            score=float(row.score),
            source="semantic",
            rank=index + 1,
            kind="catalog",
        )
        for index, row in enumerate(result)
    ]


async def _legacy_semantic_search_postgres(
    session: AsyncSession,
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[SearchHit]:
    """Handle legacy semantic search postgres.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query_embedding (list[float]):
            Embedding vector produced from the user query.
        k (int):
            Maximum number of ranked candidates or results to return.

    Returns:
        list[SearchHit]:
            Value produced for the caller according to the function contract.
    """
    vector_literal = _to_pgvector_literal(query_embedding)
    result = await session.execute(
        text(
            """
            SELECT id, 1 - (embedding <=> CAST(:query_embedding AS vector)) AS score
            FROM menu_items
            WHERE tenant_id = :tenant_id
              AND is_available = true
              AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:query_embedding AS vector), id
            LIMIT :limit
            """
        ),
        {
            "tenant_id": tenant_id,
            "query_embedding": vector_literal,
            "limit": k,
        },
    )
    return [
        SearchHit(item_id=row.id, score=float(row.score), source="semantic", rank=index + 1)
        for index, row in enumerate(result)
    ]


async def _catalog_semantic_search_python(
    session: AsyncSession,
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[SearchHit]:
    """Handle catalog semantic search python.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query_embedding (list[float]):
            Embedding vector produced from the user query.
        k (int):
            Maximum number of ranked candidates or results to return.

    Returns:
        list[SearchHit]:
            Value produced for the caller according to the function contract.
    """
    result = await session.execute(
        select(CatalogItemVariant.id, CatalogItemEmbedding.embedding)
        .join(CatalogItemEmbedding, CatalogItemEmbedding.variant_id == CatalogItemVariant.id)
        .join(CatalogItem)
        .join(MenuVersion)
        .join(Menu)
        .where(Menu.tenant_id == tenant_id)
        .where(MenuVersion.status == "published")
        .where(CatalogItem.is_available.is_(True))
        .where(CatalogItemVariant.is_available.is_(True))
        .where(CatalogItemEmbedding.embedding.is_not(None))
        .order_by(CatalogItemVariant.id)
    )
    scored: list[tuple[int, float]] = []
    for variant_id, embedding in result:
        score = cosine_similarity(query_embedding, embedding)
        if score > 0:
            scored.append((variant_id, score))

    scored.sort(key=lambda item_score: (-item_score[1], item_score[0]))
    return [
        SearchHit(
            item_id=item_id,
            score=score,
            source="semantic",
            rank=index + 1,
            kind="catalog",
        )
        for index, (item_id, score) in enumerate(scored[:k])
    ]


async def _legacy_semantic_search_python(
    session: AsyncSession,
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[SearchHit]:
    """Handle legacy semantic search python.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query_embedding (list[float]):
            Embedding vector produced from the user query.
        k (int):
            Maximum number of ranked candidates or results to return.

    Returns:
        list[SearchHit]:
            Value produced for the caller according to the function contract.
    """
    result = await session.execute(
        select(MenuItem.id, MenuItem.embedding)
        .where(MenuItem.tenant_id == tenant_id)
        .where(MenuItem.is_available.is_(True))
        .where(MenuItem.embedding.is_not(None))
        .order_by(MenuItem.id)
    )
    scored: list[tuple[int, float]] = []
    for item_id, embedding in result:
        score = cosine_similarity(query_embedding, embedding)
        if score > 0:
            scored.append((item_id, score))

    scored.sort(key=lambda item_score: (-item_score[1], item_score[0]))
    return [
        SearchHit(item_id=item_id, score=score, source="semantic", rank=index + 1)
        for index, (item_id, score) in enumerate(scored[:k])
    ]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Handle cosine similarity.

    Args:
        left (list[float]):
            Left value required to perform this operation.
        right (list[float]):
            Right value required to perform this operation.

    Returns:
        float:
            Value produced for the caller according to the function contract.
    """
    if len(left) != len(right) or not left:
        return 0.0

    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _to_pgvector_literal(vector: list[float]) -> str:
    """Convert pgvector literal.

    Args:
        vector (list[float]):
            Vector being normalized, converted, or sent to the vector store.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    return "[" + ",".join(str(float(component)) for component in vector) + "]"
