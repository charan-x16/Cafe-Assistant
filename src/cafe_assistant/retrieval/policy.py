"""Hybrid retrieval for tenant-scoped policy document chunks.

Policy retrieval is separate from menu retrieval. It searches policy chunks from
company documents so the agent can answer operational questions, but those chunks
are never treated as menu recommendations. Like menu retrieval, Qdrant is only an
index: returned IDs are resolved back to SQL before text enters any model context.
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
from cafe_assistant.observability.metrics import record_quality_event
from cafe_assistant.retrieval.embeddings import embed_query
from cafe_assistant.retrieval.qdrant_store import (
    QdrantVectorStoreError,
    qdrant_enabled,
    search_policy_chunk_vectors,
)
from cafe_assistant.retrieval.vector_store import cosine_similarity

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_RRF_K = 60


@dataclass(frozen=True, slots=True)
class PolicyChunkResult:
    """Authoritative policy chunk returned by policy retrieval.

    The result contains SQL-loaded text and ranking metadata. It is safe for
    grounded policy answering, but it is not a menu item and must not be passed
    into recommendation-only contexts.
    """

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
    """Retrieve policy chunks with keyword/semantic rank fusion.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used to search and reload policy chunks.
        tenant_id (int):
            Tenant ID used to scope every keyword, vector, and SQL reload query.
        query (str):
            User policy question or search text.
        k (int):
            Maximum number of final policy chunks to return.
        embedding_provider (EmbeddingProvider | None):
            Optional embedding provider override used by tests. When omitted, the
            configured provider from the model gateway is used.

    Returns:
        list[PolicyChunkResult]:
            Ranked policy chunks loaded from SQL with heading path, content,
            fused score, and one-based rank. Empty input or no matches returns an
            empty list.
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
    """Find policy chunks using deterministic token and fuzzy text matching.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used to load tenant policy chunks.
        tenant_id (int):
            Tenant ID joined through `policy_documents`.
        query (str):
            User policy question or search text.
        k (int):
            Maximum number of keyword candidates to return.

    Returns:
        list[tuple[int, float]]:
            Policy chunk IDs paired with deterministic keyword scores, sorted by
            descending score and then ascending chunk ID.
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
    """Search policy embeddings with Qdrant, pgvector, or Python fallback.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for SQL/pgvector fallback reads.
        tenant_id (int):
            Tenant ID applied to Qdrant filters and SQL joins.
        query_embedding (list[float]):
            Query vector produced by the configured embedding provider/model.
        k (int):
            Maximum number of semantic policy candidates to return.

    Returns:
        list[tuple[int, float]]:
            Policy chunk IDs paired with semantic similarity scores. If Qdrant is
            configured but unavailable, the function records fallback metrics and
            uses the SQL/pgvector path instead of failing the request.
    """
    bind = session.get_bind()
    if bind.dialect.name == "postgresql":
        if qdrant_enabled():
            try:
                qdrant_hits = await search_policy_chunk_vectors(tenant_id, query_embedding, k)
                if qdrant_hits:
                    return qdrant_hits
            except QdrantVectorStoreError:
                record_quality_event(
                    "retrieval_qdrant_failures_total",
                    source_kind="policy_chunk",
                )
                record_quality_event(
                    "retrieval_semantic_fallback_total",
                    source_kind="policy_chunk",
                    fallback="pgvector",
                )
        return await _policy_semantic_search_postgres(session, tenant_id, query_embedding, k)
    return await _policy_semantic_search_python(session, tenant_id, query_embedding, k)


async def _policy_semantic_search_postgres(
    session: AsyncSession,
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[tuple[int, float]]:
    """Search policy chunk embeddings stored in Postgres pgvector.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session bound to Postgres.
        tenant_id (int):
            Tenant ID joined through `policy_documents`.
        query_embedding (list[float]):
            Query vector serialized into pgvector literal format.
        k (int):
            Maximum number of policy chunk hits to return.

    Returns:
        list[tuple[int, float]]:
            Policy chunk IDs and pgvector similarity scores ordered by vector
            distance and chunk ID.
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
    """Search policy embeddings in Python for SQLite-based tests.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session bound to SQLite or another non-Postgres DB.
        tenant_id (int):
            Tenant ID joined through `policy_documents`.
        query_embedding (list[float]):
            Query vector compared with stored Python lists.
        k (int):
            Maximum number of policy chunk hits to return.

    Returns:
        list[tuple[int, float]]:
            Policy chunk IDs and cosine-similarity scores ordered by descending
            score and ascending chunk ID.
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
    """Score one policy chunk against a query using deterministic text signals.

    Args:
        query (str):
            Original user policy query.
        query_tokens (set[str]):
            Lowercased alphanumeric query tokens.
        chunk (PolicyChunk):
            Policy chunk row whose heading path and content are scored.

    Returns:
        float:
            Positive score when the query exactly, token-wise, or fuzzily matches
            the chunk; zero when no useful keyword signal is found.
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
    """Tokenize policy text into lowercased alphanumeric terms.

    Args:
        text_value (str):
            Raw query, heading, or policy content to tokenize.

    Returns:
        set[str]:
            Unique lowercase tokens used by deterministic keyword scoring.
    """
    return set(_TOKEN_PATTERN.findall(text_value.lower()))