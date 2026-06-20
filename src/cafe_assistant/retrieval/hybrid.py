"""Implementation module for hybrid.
Contains typed helpers used by the cafe assistant backend runtime.
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.repositories.menu_repo import (
    load_menu_item_views_for_tenant,
    load_published_catalog_item_views_for_tenant,
)
from cafe_assistant.domain.dietary import CustomerRestrictions, MenuItemView, filter_safe_items
from cafe_assistant.gateway.model_gateway import EmbeddingProvider, get_embedding_provider
from cafe_assistant.retrieval.embeddings import embed_query
from cafe_assistant.retrieval.keyword import keyword_search
from cafe_assistant.retrieval.types import SearchHit
from cafe_assistant.retrieval.vector_store import semantic_search

RRF_K = 60


async def hybrid_search(
    session: AsyncSession,
    tenant_id: int,
    query: str,
    k: int,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[SearchHit]:
    """Fuse keyword and semantic menu hits into one ranked candidate list.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query (str):
            User search text or policy question to retrieve against.
        k (int):
            Maximum number of ranked candidates or results to return.
        embedding_provider (EmbeddingProvider | None):
            Embedding provider used to create query or record vectors.

    Returns:
        list[SearchHit]:
            Ranked search hits fused from keyword and semantic retrieval.
    """
    if k <= 0 or not query.strip():
        return []

    provider = embedding_provider or get_embedding_provider()
    query_embedding = embed_query(query, provider)
    candidate_limit = max(k * 4, 20)

    keyword_hits = await keyword_search(session, tenant_id, query, candidate_limit)
    semantic_hits = await semantic_search(session, tenant_id, query_embedding, candidate_limit)

    fused_scores: dict[tuple[str, int], float] = defaultdict(float)
    best_source_scores: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for hit in [*keyword_hits, *semantic_hits]:
        key = (hit.kind, hit.item_id)
        fused_scores[key] += 1.0 / (RRF_K + hit.rank)
        best_source_scores[key][hit.source] = max(
            best_source_scores[key].get(hit.source, 0.0),
            hit.score,
        )

    reranked = [
        SearchHit(
            item_id=item_id,
            score=_rerank_score(fused_score, best_source_scores[(kind, item_id)]),
            source="hybrid",
            rank=0,
            kind=kind,
        )
        for (kind, item_id), fused_score in fused_scores.items()
    ]
    reranked.sort(key=lambda hit: (-hit.score, hit.kind, hit.item_id))
    return [
        SearchHit(
            item_id=hit.item_id,
            score=hit.score,
            source=hit.source,
            rank=index + 1,
            kind=hit.kind,
        )
        for index, hit in enumerate(reranked[:candidate_limit])
    ]


async def search_menu(
    session: AsyncSession,
    tenant_id: int,
    query: str,
    restrictions: CustomerRestrictions,
    k: int = 10,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[MenuItemView]:
    """Retrieve menu candidates and return only items approved by the safety filter.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query (str):
            User search text or policy question to retrieve against.
        restrictions (CustomerRestrictions):
            Customer allergen, dietary, and sugar preferences for the active turn.
        k (int):
            Maximum number of ranked candidates or results to return.
        embedding_provider (EmbeddingProvider | None):
            Embedding provider used to create query or record vectors.

    Returns:
        list[MenuItemView]:
            Safe menu item views after retrieval and deterministic filtering.
    """
    candidates = await hybrid_search(session, tenant_id, query, k, embedding_provider)
    catalog_ids = [hit.item_id for hit in candidates if hit.kind == "catalog"]
    legacy_ids = [hit.item_id for hit in candidates if hit.kind == "legacy"]

    catalog_views = await load_published_catalog_item_views_for_tenant(
        session,
        tenant_id,
        catalog_ids,
    )
    legacy_views = await load_menu_item_views_for_tenant(session, tenant_id, legacy_ids)
    views_by_key = {
        **{("catalog", view.id): view for view in catalog_views},
        **{("legacy", view.id): view for view in legacy_views},
    }
    ranked_views = [
        views_by_key[(hit.kind, hit.item_id)]
        for hit in candidates
        if (hit.kind, hit.item_id) in views_by_key
    ]

    return filter_safe_items(ranked_views, restrictions).safe_items[:k]


def _rerank_score(fused_score: float, source_scores: dict[str, float]) -> float:
    """Handle rerank score.

    Args:
        fused_score (float):
            Fused score value required to perform this operation.
        source_scores (dict[str, float]):
            Source scores value required to perform this operation.

    Returns:
        float:
            Value produced for the caller according to the function contract.
    """
    keyword_component = source_scores.get("keyword", 0.0)
    semantic_component = source_scores.get("semantic", 0.0)
    return fused_score + (0.05 * keyword_component) + (0.05 * semantic_component)
