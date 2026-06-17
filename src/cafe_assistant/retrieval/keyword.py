from __future__ import annotations

import re
from difflib import SequenceMatcher

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cafe_assistant.db.models import Ingredient, MenuItem
from cafe_assistant.retrieval.types import SearchHit

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


async def keyword_search(
    session: AsyncSession,
    tenant_id: int,
    query: str,
    k: int,
) -> list[SearchHit]:
    if k <= 0 or not query.strip():
        return []

    bind = session.get_bind()
    if bind.dialect.name == "postgresql":
        return await _keyword_search_postgres(session, tenant_id, query, k)
    return await _keyword_search_python(session, tenant_id, query, k)


async def _keyword_search_postgres(
    session: AsyncSession,
    tenant_id: int,
    query: str,
    k: int,
) -> list[SearchHit]:
    result = await session.execute(
        text(
            """
            WITH query AS (
                SELECT
                    plainto_tsquery('english', :query_text) AS tsq,
                    lower(:query_text) AS raw_query
            )
            SELECT
                menu_items.id,
                (
                    ts_rank_cd(
                        to_tsvector(
                            'english',
                            coalesce(menu_items.name, '') || ' ' ||
                            coalesce(menu_items.description, '') || ' ' ||
                            coalesce(menu_items.category, '')
                        ),
                        query.tsq
                    )
                    + similarity(lower(menu_items.name), query.raw_query)
                    + CASE WHEN lower(menu_items.name) = query.raw_query THEN 2.0 ELSE 0.0 END
                    + CASE WHEN lower(menu_items.name) LIKE '%' || query.raw_query || '%'
                           THEN 0.75 ELSE 0.0 END
                ) AS score
            FROM menu_items, query
            WHERE menu_items.tenant_id = :tenant_id
              AND menu_items.is_available = true
              AND (
                    to_tsvector(
                        'english',
                        coalesce(menu_items.name, '') || ' ' ||
                        coalesce(menu_items.description, '') || ' ' ||
                        coalesce(menu_items.category, '')
                    ) @@ query.tsq
                    OR lower(menu_items.name) % query.raw_query
                    OR lower(menu_items.description) % query.raw_query
                    OR lower(menu_items.name) LIKE '%' || query.raw_query || '%'
              )
            ORDER BY score DESC, menu_items.id
            LIMIT :limit
            """
        ),
        {
            "tenant_id": tenant_id,
            "query_text": query,
            "limit": k,
        },
    )
    return [
        SearchHit(item_id=row.id, score=float(row.score), source="keyword", rank=index + 1)
        for index, row in enumerate(result)
    ]


async def _keyword_search_python(
    session: AsyncSession,
    tenant_id: int,
    query: str,
    k: int,
) -> list[SearchHit]:
    result = await session.scalars(
        select(MenuItem)
        .where(MenuItem.tenant_id == tenant_id)
        .where(MenuItem.is_available.is_(True))
        .options(
            selectinload(MenuItem.dietary_tags),
            selectinload(MenuItem.ingredients).selectinload(Ingredient.allergens),
        )
        .order_by(MenuItem.id)
    )
    query_tokens = _tokens(query)
    scored: list[tuple[int, float]] = []
    for item in result.unique():
        score = _keyword_score(query, query_tokens, item)
        if score > 0:
            scored.append((item.id, score))

    scored.sort(key=lambda item_score: (-item_score[1], item_score[0]))
    return [
        SearchHit(item_id=item_id, score=score, source="keyword", rank=index + 1)
        for index, (item_id, score) in enumerate(scored[:k])
    ]


def _keyword_score(query: str, query_tokens: set[str], item: MenuItem) -> float:
    name = item.name.lower()
    fields = " ".join(
        (
            item.name,
            item.description,
            item.category,
            " ".join(tag.code for tag in item.dietary_tags),
            " ".join(ingredient.name for ingredient in item.ingredients),
        )
    ).lower()
    field_tokens = _tokens(fields)
    score = 0.0

    if query.lower() == name:
        score += 10.0
    if query.lower() in name:
        score += 4.0
    if query.lower() in fields:
        score += 1.5

    for token in query_tokens:
        if token in field_tokens:
            score += 2.0
            continue
        fuzzy_match = max(
            (SequenceMatcher(None, token, field_token).ratio() for field_token in field_tokens),
            default=0.0,
        )
        if fuzzy_match >= 0.78:
            score += fuzzy_match

    return score


def _tokens(text_value: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text_value.lower()))
