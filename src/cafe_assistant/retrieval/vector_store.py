"""Semantic menu retrieval over Qdrant, pgvector, and local test fallbacks.

This module returns ranked candidate IDs only. It does not decide dietary or
allergen safety and it does not expose raw menu content to the LLM. The public
`semantic_search` function first tries the configured production vector index,
then falls back to SQL/pgvector when Qdrant is unavailable, and finally uses a
pure Python cosine path for SQLite-based tests.
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
from cafe_assistant.observability.metrics import record_quality_event
from cafe_assistant.retrieval.qdrant_store import (
    QdrantVectorStoreError,
    qdrant_enabled,
    search_catalog_item_vectors,
)
from cafe_assistant.retrieval.types import SearchHit


async def semantic_search(
    session: AsyncSession,
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[SearchHit]:
    """Return tenant-scoped menu candidates ranked by embedding similarity.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for SQL/pgvector fallback reads.
        tenant_id (int):
            Tenant ID applied to every vector-store and SQL query.
        query_embedding (list[float]):
            Query vector produced by the configured embedding provider/model.
        k (int):
            Maximum number of semantic candidates to return.

    Returns:
        list[SearchHit]:
            Ranked semantic hits. Catalog hits contain catalog variant IDs and
            `kind="catalog"`; legacy hits contain legacy `menu_items.id` values.
            Callers must reload authoritative SQL rows and run the safety filter
            before returning anything to the agent or user.
    """
    if k <= 0:
        return []

    bind = session.get_bind()
    if bind.dialect.name == "postgresql":
        if qdrant_enabled():
            try:
                qdrant_hits = await search_catalog_item_vectors(tenant_id, query_embedding, k)
                if qdrant_hits:
                    return qdrant_hits
            except QdrantVectorStoreError:
                record_quality_event(
                    "retrieval_qdrant_failures_total",
                    source_kind="catalog_item",
                )
                record_quality_event(
                    "retrieval_semantic_fallback_total",
                    source_kind="catalog_item",
                    fallback="pgvector",
                )
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
    """Search catalog variant embeddings stored in Postgres pgvector.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session bound to Postgres.
        tenant_id (int):
            Tenant ID joined through menus and menu versions.
        query_embedding (list[float]):
            Query vector converted to a pgvector literal.
        k (int):
            Maximum number of catalog variant hits to return.

    Returns:
        list[SearchHit]:
            Catalog semantic hits ordered by cosine distance, each carrying a
            catalog variant ID and `kind="catalog"`.
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
    """Search legacy `menu_items.embedding` values with pgvector.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session bound to Postgres.
        tenant_id (int):
            Tenant ID used to scope legacy menu rows.
        query_embedding (list[float]):
            Query vector converted to a pgvector literal.
        k (int):
            Maximum number of legacy menu item hits to return.

    Returns:
        list[SearchHit]:
            Legacy semantic hits ordered by cosine distance. These are returned
            only when catalog embeddings are unavailable for the tenant.
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
    """Search catalog embeddings in Python for SQLite-based tests.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session bound to SQLite or another non-Postgres DB.
        tenant_id (int):
            Tenant ID joined through menus and menu versions.
        query_embedding (list[float]):
            Query vector compared with stored Python lists.
        k (int):
            Maximum number of catalog variant hits to return.

    Returns:
        list[SearchHit]:
            Catalog semantic hits ranked by local cosine similarity. This path is
            intended for deterministic tests and local fallback behavior.
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
    """Search legacy menu embeddings in Python for non-Postgres tests.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session bound to SQLite or another non-Postgres DB.
        tenant_id (int):
            Tenant ID used to scope legacy menu rows.
        query_embedding (list[float]):
            Query vector compared with stored Python lists.
        k (int):
            Maximum number of legacy menu hits to return.

    Returns:
        list[SearchHit]:
            Legacy semantic hits ranked by local cosine similarity.
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
    """Compute cosine similarity for two equal-length vectors.

    Args:
        left (list[float]):
            First vector, usually the query embedding.
        right (list[float]):
            Second vector, usually a stored menu or catalog embedding.

    Returns:
        float:
            Cosine similarity in the range -1.0 to 1.0 for valid vectors. Returns
            0.0 when vectors are empty, have different lengths, or have zero norm.
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
    """Serialize a Python vector into pgvector's bracketed literal format.

    Args:
        vector (list[float]):
            Embedding vector to pass as a bound SQL parameter and cast to
            Postgres `vector`.

    Returns:
        str:
            String such as `[0.1,0.2,0.3]` that pgvector can parse after
            `CAST(:query_embedding AS vector)`.
    """
    return "[" + ",".join(str(float(component)) for component in vector) + "]"