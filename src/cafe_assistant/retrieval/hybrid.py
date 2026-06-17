from __future__ import annotations

from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.repositories.menu_repo import load_menu_item_views_for_tenant
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
    if k <= 0 or not query.strip():
        return []

    provider = embedding_provider or get_embedding_provider()
    query_embedding = embed_query(query, provider)
    candidate_limit = max(k * 4, 20)

    keyword_hits = await keyword_search(session, tenant_id, query, candidate_limit)
    semantic_hits = await semantic_search(session, tenant_id, query_embedding, candidate_limit)

    fused_scores: dict[int, float] = defaultdict(float)
    best_source_scores: dict[int, dict[str, float]] = defaultdict(dict)
    for hit in [*keyword_hits, *semantic_hits]:
        fused_scores[hit.item_id] += 1.0 / (RRF_K + hit.rank)
        best_source_scores[hit.item_id][hit.source] = max(
            best_source_scores[hit.item_id].get(hit.source, 0.0),
            hit.score,
        )

    reranked = [
        SearchHit(
            item_id=item_id,
            score=_rerank_score(fused_score, best_source_scores[item_id]),
            source="hybrid",
            rank=0,
        )
        for item_id, fused_score in fused_scores.items()
    ]
    reranked.sort(key=lambda hit: (-hit.score, hit.item_id))
    return [
        SearchHit(item_id=hit.item_id, score=hit.score, source=hit.source, rank=index + 1)
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
    candidates = await hybrid_search(session, tenant_id, query, k, embedding_provider)
    candidate_ids = [hit.item_id for hit in candidates]
    views = await load_menu_item_views_for_tenant(session, tenant_id, candidate_ids)
    views_by_id = {view.id: view for view in views}
    ranked_views = [views_by_id[item_id] for item_id in candidate_ids if item_id in views_by_id]

    return filter_safe_items(ranked_views, restrictions).safe_items[:k]


def _rerank_score(fused_score: float, source_scores: dict[str, float]) -> float:
    keyword_component = source_scores.get("keyword", 0.0)
    semantic_component = source_scores.get("semantic", 0.0)
    return fused_score + (0.05 * keyword_component) + (0.05 * semantic_component)
