from __future__ import annotations

import math

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.models import MenuItem
from cafe_assistant.retrieval.types import SearchHit


async def semantic_search(
    session: AsyncSession,
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[SearchHit]:
    if k <= 0:
        return []

    bind = session.get_bind()
    if bind.dialect.name == "postgresql":
        return await _semantic_search_postgres(session, tenant_id, query_embedding, k)
    return await _semantic_search_python(session, tenant_id, query_embedding, k)


async def _semantic_search_postgres(
    session: AsyncSession,
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[SearchHit]:
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


async def _semantic_search_python(
    session: AsyncSession,
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[SearchHit]:
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
        scored.append((item_id, score))

    scored.sort(key=lambda item_score: (-item_score[1], item_score[0]))
    return [
        SearchHit(item_id=item_id, score=score, source="semantic", rank=index + 1)
        for index, (item_id, score) in enumerate(scored[:k])
    ]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0

    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _to_pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(float(component)) for component in vector) + "]"
