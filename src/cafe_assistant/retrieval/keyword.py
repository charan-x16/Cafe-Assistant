"""Implementation module for keyword.
Contains typed helpers used by the cafe assistant backend runtime.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cafe_assistant.db.models import (
    CatalogItem,
    CatalogItemAllergenAssertion,
    CatalogItemDietaryAssertion,
    CatalogItemVariant,
    Ingredient,
    Menu,
    MenuItem,
    MenuVersion,
)
from cafe_assistant.retrieval.types import SearchHit

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


async def keyword_search(
    session: AsyncSession,
    tenant_id: int,
    query: str,
    k: int,
) -> list[SearchHit]:
    """Search menu records with exact, keyword, and fuzzy text signals.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query (str):
            User search text or policy question to retrieve against.
        k (int):
            Maximum number of ranked candidates or results to return.

    Returns:
        list[SearchHit]:
            Ranked keyword/fuzzy hits for catalog or legacy menu records.
    """
    if k <= 0 or not query.strip():
        return []

    bind = session.get_bind()
    if bind.dialect.name == "postgresql":
        catalog_hits = await _catalog_keyword_search_postgres(session, tenant_id, query, k)
        if catalog_hits:
            return catalog_hits
        return await _legacy_keyword_search_postgres(session, tenant_id, query, k)

    catalog_hits = await _catalog_keyword_search_python(session, tenant_id, query, k)
    if catalog_hits:
        return catalog_hits
    return await _legacy_keyword_search_python(session, tenant_id, query, k)


async def _catalog_keyword_search_postgres(
    session: AsyncSession,
    tenant_id: int,
    query: str,
    k: int,
) -> list[SearchHit]:
    """Handle catalog keyword search postgres.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query (str):
            User search text or policy question to retrieve against.
        k (int):
            Maximum number of ranked candidates or results to return.

    Returns:
        list[SearchHit]:
            Value produced for the caller according to the function contract.
    """
    result = await session.execute(
        text(
            """
            WITH query AS (
                SELECT
                    plainto_tsquery('english', :query_text) AS tsq,
                    lower(:query_text) AS raw_query
            ),
            catalog AS (
                SELECT
                    catalog_item_variants.id,
                    catalog_items.display_name,
                    catalog_item_variants.name AS variant_name,
                    catalog_items.description,
                    coalesce(menu_categories.path, '') AS category_path,
                    coalesce(catalog_item_variants.serving, '') AS serving,
                    coalesce(catalog_item_variants.temperature, '') AS temperature,
                    coalesce(catalog_items.tags::text, '') AS tags
                FROM catalog_item_variants
                JOIN catalog_items
                    ON catalog_items.id = catalog_item_variants.catalog_item_id
                JOIN menu_versions
                    ON menu_versions.id = catalog_items.menu_version_id
                JOIN menus
                    ON menus.id = menu_versions.menu_id
                LEFT JOIN menu_categories
                    ON menu_categories.id = catalog_items.category_id
                WHERE menus.tenant_id = :tenant_id
                  AND menu_versions.status = 'published'
                  AND catalog_items.is_available = true
                  AND catalog_item_variants.is_available = true
            )
            SELECT
                catalog.id,
                (
                    ts_rank_cd(
                        to_tsvector(
                            'english',
                            coalesce(catalog.display_name, '') || ' ' ||
                            coalesce(catalog.variant_name, '') || ' ' ||
                            coalesce(catalog.description, '') || ' ' ||
                            coalesce(catalog.category_path, '') || ' ' ||
                            coalesce(catalog.serving, '') || ' ' ||
                            coalesce(catalog.temperature, '') || ' ' ||
                            coalesce(catalog.tags, '')
                        ),
                        query.tsq
                    )
                    + similarity(lower(catalog.display_name), query.raw_query)
                    + CASE WHEN lower(catalog.display_name) = query.raw_query
                           THEN 2.0 ELSE 0.0 END
                    + CASE WHEN lower(catalog.display_name) LIKE '%' || query.raw_query || '%'
                           THEN 0.75 ELSE 0.0 END
                ) AS score
            FROM catalog, query
            WHERE (
                    to_tsvector(
                        'english',
                        coalesce(catalog.display_name, '') || ' ' ||
                        coalesce(catalog.variant_name, '') || ' ' ||
                        coalesce(catalog.description, '') || ' ' ||
                        coalesce(catalog.category_path, '') || ' ' ||
                        coalesce(catalog.serving, '') || ' ' ||
                        coalesce(catalog.temperature, '') || ' ' ||
                        coalesce(catalog.tags, '')
                    ) @@ query.tsq
                    OR lower(catalog.display_name) % query.raw_query
                    OR lower(catalog.description) % query.raw_query
                    OR lower(catalog.display_name) LIKE '%' || query.raw_query || '%'
              )
            ORDER BY score DESC, catalog.id
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
        SearchHit(
            item_id=row.id,
            score=float(row.score),
            source="keyword",
            rank=index + 1,
            kind="catalog",
        )
        for index, row in enumerate(result)
    ]


async def _legacy_keyword_search_postgres(
    session: AsyncSession,
    tenant_id: int,
    query: str,
    k: int,
) -> list[SearchHit]:
    """Handle legacy keyword search postgres.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query (str):
            User search text or policy question to retrieve against.
        k (int):
            Maximum number of ranked candidates or results to return.

    Returns:
        list[SearchHit]:
            Value produced for the caller according to the function contract.
    """
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


async def _catalog_keyword_search_python(
    session: AsyncSession,
    tenant_id: int,
    query: str,
    k: int,
) -> list[SearchHit]:
    """Handle catalog keyword search python.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query (str):
            User search text or policy question to retrieve against.
        k (int):
            Maximum number of ranked candidates or results to return.

    Returns:
        list[SearchHit]:
            Value produced for the caller according to the function contract.
    """
    result = await session.scalars(
        select(CatalogItemVariant)
        .join(CatalogItem)
        .join(MenuVersion)
        .join(Menu)
        .where(Menu.tenant_id == tenant_id)
        .where(MenuVersion.status == "published")
        .where(CatalogItem.is_available.is_(True))
        .where(CatalogItemVariant.is_available.is_(True))
        .options(
            selectinload(CatalogItemVariant.catalog_item).selectinload(CatalogItem.category),
            selectinload(CatalogItemVariant.catalog_item).selectinload(CatalogItem.ingredients),
            selectinload(CatalogItemVariant.catalog_item)
            .selectinload(CatalogItem.allergen_assertions)
            .selectinload(CatalogItemAllergenAssertion.allergen),
            selectinload(CatalogItemVariant.catalog_item)
            .selectinload(CatalogItem.dietary_assertions)
            .selectinload(CatalogItemDietaryAssertion.dietary_tag),
        )
        .order_by(CatalogItem.sort_order, CatalogItemVariant.sort_order, CatalogItemVariant.id)
    )
    query_tokens = _tokens(query)
    scored: list[tuple[int, float]] = []
    for variant in result.unique():
        score = _catalog_keyword_score(query, query_tokens, variant)
        if score > 0:
            scored.append((variant.id, score))

    scored.sort(key=lambda item_score: (-item_score[1], item_score[0]))
    return [
        SearchHit(
            item_id=item_id,
            score=score,
            source="keyword",
            rank=index + 1,
            kind="catalog",
        )
        for index, (item_id, score) in enumerate(scored[:k])
    ]


async def _legacy_keyword_search_python(
    session: AsyncSession,
    tenant_id: int,
    query: str,
    k: int,
) -> list[SearchHit]:
    """Handle legacy keyword search python.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query (str):
            User search text or policy question to retrieve against.
        k (int):
            Maximum number of ranked candidates or results to return.

    Returns:
        list[SearchHit]:
            Value produced for the caller according to the function contract.
    """
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
    """Handle keyword score.

    Args:
        query (str):
            User search text or policy question to retrieve against.
        query_tokens (set[str]):
            Query tokens value required to perform this operation.
        item (MenuItem):
            Menu or catalog item being transformed, embedded, or evaluated.

    Returns:
        float:
            Value produced for the caller according to the function contract.
    """
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


def _catalog_keyword_score(
    query: str,
    query_tokens: set[str],
    variant: CatalogItemVariant,
) -> float:
    """Handle catalog keyword score.

    Args:
        query (str):
            User search text or policy question to retrieve against.
        query_tokens (set[str]):
            Query tokens value required to perform this operation.
        variant (CatalogItemVariant):
            Catalog variant row converted into a retrievable menu view.

    Returns:
        float:
            Value produced for the caller according to the function contract.
    """
    item = variant.catalog_item
    display_name = item.display_name.lower()
    category_path = item.category.path if item.category is not None else ""
    fields = " ".join(
        (
            item.display_name,
            "" if variant.name == "Default" else variant.name,
            item.description,
            category_path,
            variant.serving or "",
            variant.temperature or "",
            variant.caffeine_level or "",
            variant.sweetness_level or "",
            variant.spice_level or "",
            " ".join(item.tags),
            " ".join(ingredient.name for ingredient in item.ingredients),
            " ".join(assertion.allergen.code for assertion in item.allergen_assertions),
            " ".join(assertion.dietary_tag.code for assertion in item.dietary_assertions),
        )
    ).lower()
    field_tokens = _tokens(fields)
    score = 0.0

    if query.lower() == display_name:
        score += 10.0
    if query.lower() in display_name:
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
    """Handle tokens.

    Args:
        text_value (str):
            Raw text being parsed, cleaned, hashed, or tokenized.

    Returns:
        set[str]:
            Value produced for the caller according to the function contract.
    """
    return set(_TOKEN_PATTERN.findall(text_value.lower()))
