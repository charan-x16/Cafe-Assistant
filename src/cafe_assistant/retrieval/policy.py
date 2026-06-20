"""Implementation module for policy.
Contains typed helpers used by the cafe assistant backend runtime.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.models import PolicyChunk, PolicyChunkEmbedding, PolicyDocument
from cafe_assistant.gateway.model_gateway import EmbeddingProvider, get_embedding_provider
from cafe_assistant.retrieval.embeddings import embed_query
from cafe_assistant.retrieval.qdrant_store import qdrant_enabled, search_policy_chunk_vectors
from cafe_assistant.retrieval.vector_store import cosine_similarity

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_RRF_K = 60


@dataclass(frozen=True, slots=True)
class PolicyChunkResult:
    """Container for policy chunk result behavior and data."""
    id: int
    heading_path: str
    content: str
    score: float
    rank: int


async def search_policy_chunks(
    session: AsyncSession,
    tenant_id: int,
    query: str,
    k: int = 5,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[PolicyChunkResult]:
    """Retrieve policy chunks with fused keyword and semantic ranking.

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
        list[PolicyChunkResult]:
            Ranked policy chunks with authoritative text loaded from SQL.
    """
    if k <= 0 or not query.strip():
        return []

    provider = embedding_provider or get_embedding_provider()
    query_embedding = embed_query(query, provider)
    candidate_limit = max(k * 4, 20)
    keyword_hits = await _policy_keyword_search(session, tenant_id, query, candidate_limit)
    semantic_hits = await _policy_semantic_search(
        session,
        tenant_id,
        query_embedding,
        candidate_limit,
    )

    fused_scores: dict[int, float] = defaultdict(float)
    for rank, (chunk_id, score) in enumerate(keyword_hits, start=1):
        fused_scores[chunk_id] += (1.0 / (_RRF_K + rank)) + (0.05 * score)
    for rank, (chunk_id, score) in enumerate(semantic_hits, start=1):
        fused_scores[chunk_id] += (1.0 / (_RRF_K + rank)) + (0.05 * score)

    if not fused_scores:
        return []

    chunks = await session.scalars(
        select(PolicyChunk)
        .join(PolicyDocument)
        .where(PolicyDocument.tenant_id == tenant_id)
        .where(PolicyChunk.id.in_(fused_scores))
    )
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    ranked_ids = sorted(fused_scores, key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id))
    return [
        PolicyChunkResult(
            id=chunk_id,
            heading_path=chunks_by_id[chunk_id].heading_path,
            content=chunks_by_id[chunk_id].content,
            score=fused_scores[chunk_id],
            rank=rank,
        )
        for rank, chunk_id in enumerate(ranked_ids[:k], start=1)
        if chunk_id in chunks_by_id
    ]


async def _policy_keyword_search(
    session: AsyncSession,
    tenant_id: int,
    query: str,
    k: int,
) -> list[tuple[int, float]]:
    """Handle policy keyword search.

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
        list[tuple[int, float]]:
            Value produced for the caller according to the function contract.
    """
    chunks = await session.scalars(
        select(PolicyChunk)
        .join(PolicyDocument)
        .where(PolicyDocument.tenant_id == tenant_id)
        .order_by(PolicyDocument.id, PolicyChunk.chunk_index)
    )
    query_tokens = _tokens(query)
    scored: list[tuple[int, float]] = []
    for chunk in chunks:
        score = _keyword_score(query, query_tokens, chunk)
        if score > 0:
            scored.append((chunk.id, score))
    scored.sort(key=lambda item_score: (-item_score[1], item_score[0]))
    return scored[:k]


async def _policy_semantic_search(
    session: AsyncSession,
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[tuple[int, float]]:
    """Handle policy semantic search.

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
        list[tuple[int, float]]:
            Value produced for the caller according to the function contract.
    """
    bind = session.get_bind()
    if bind.dialect.name == "postgresql" and qdrant_enabled():
        try:
            return await search_policy_chunk_vectors(tenant_id, query_embedding, k)
        except RuntimeError:
            return []

    if bind.dialect.name == "postgresql":
        return await _policy_semantic_search_postgres(session, tenant_id, query_embedding, k)
    return await _policy_semantic_search_python(session, tenant_id, query_embedding, k)


async def _policy_semantic_search_postgres(
    session: AsyncSession,
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[tuple[int, float]]:
    """Handle policy semantic search postgres.

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
        list[tuple[int, float]]:
            Value produced for the caller according to the function contract.
    """
    vector_literal = "[" + ",".join(str(float(component)) for component in query_embedding) + "]"
    result = await session.execute(
        text(
            """
            SELECT
                policy_chunks.id,
                1 - (policy_chunk_embeddings.embedding <=> CAST(:query_embedding AS vector))
                    AS score
            FROM policy_chunk_embeddings
            JOIN policy_chunks
                ON policy_chunks.id = policy_chunk_embeddings.policy_chunk_id
            JOIN policy_documents
                ON policy_documents.id = policy_chunks.policy_document_id
            WHERE policy_documents.tenant_id = :tenant_id
              AND policy_chunk_embeddings.embedding IS NOT NULL
            ORDER BY policy_chunk_embeddings.embedding <=> CAST(:query_embedding AS vector),
                     policy_chunks.id
            LIMIT :limit
            """
        ),
        {
            "tenant_id": tenant_id,
            "query_embedding": vector_literal,
            "limit": k,
        },
    )
    return [(row.id, float(row.score)) for row in result]


async def _policy_semantic_search_python(
    session: AsyncSession,
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[tuple[int, float]]:
    """Handle policy semantic search python.

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
        list[tuple[int, float]]:
            Value produced for the caller according to the function contract.
    """
    result = await session.execute(
        select(PolicyChunk.id, PolicyChunkEmbedding.embedding)
        .join(PolicyChunkEmbedding, PolicyChunkEmbedding.policy_chunk_id == PolicyChunk.id)
        .join(PolicyDocument)
        .where(PolicyDocument.tenant_id == tenant_id)
        .where(PolicyChunkEmbedding.embedding.is_not(None))
        .order_by(PolicyChunk.id)
    )
    scored: list[tuple[int, float]] = []
    for chunk_id, embedding in result:
        score = cosine_similarity(query_embedding, embedding)
        if score > 0:
            scored.append((chunk_id, score))
    scored.sort(key=lambda item_score: (-item_score[1], item_score[0]))
    return scored[:k]


def _keyword_score(query: str, query_tokens: set[str], chunk: PolicyChunk) -> float:
    """Handle keyword score.

    Args:
        query (str):
            User search text or policy question to retrieve against.
        query_tokens (set[str]):
            Query tokens value required to perform this operation.
        chunk (PolicyChunk):
            Chunk value required to perform this operation.

    Returns:
        float:
            Value produced for the caller according to the function contract.
    """
    fields = f"{chunk.heading_path} {chunk.content}".lower()
    field_tokens = _tokens(fields)
    score = 0.0

    if query.lower() in fields:
        score += 3.0
    for token in query_tokens:
        if token in field_tokens:
            score += 2.0
            continue
        fuzzy_match = max(
            (SequenceMatcher(None, token, field_token).ratio() for field_token in field_tokens),
            default=0.0,
        )
        if fuzzy_match >= 0.82:
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
