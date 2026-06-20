"""Qdrant vector index adapter for catalog item and policy chunk pointers.
Qdrant stores vectors plus tenant/source metadata only; retrieval resolves source IDs back to
Postgres/Neon for authoritative text and safety fields.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

from cafe_assistant.config import settings
from cafe_assistant.retrieval.types import SearchHit


class QdrantSourceKind(StrEnum):
    """Enumeration of supported Qdrant source kind values."""
    CATALOG_ITEM = "catalog_item"
    POLICY_CHUNK = "policy_chunk"


@dataclass(frozen=True, slots=True)
class QdrantVectorPoint:
    """Container for Qdrant vector point behavior and data."""
    tenant_id: int
    source_kind: QdrantSourceKind
    source_id: int
    vector: list[float]
    content_hash: str
    menu_version_id: int | None = None
    policy_document_id: int | None = None


def qdrant_enabled() -> bool:
    """Report whether Qdrant is the active configured vector index.

    Args:
        None.

    Returns:
        bool:
            True when vector search should use Qdrant; otherwise false.
    """
    return settings.vector_provider == "qdrant" and bool(settings.qdrant_url)


async def ensure_qdrant_collection() -> None:
    """Create or validate the configured Qdrant collection.

    Args:
        None.

    Returns:
        None:
            No value; raises if the existing collection is incompatible.
    """
    async with _client() as client:
        response = await client.get(_collection_url())
        if response.status_code == 404:
            await _create_collection(client)
            return
        _raise_for_qdrant_error(response)
        _validate_collection_config(response.json())


async def upsert_qdrant_points(points: list[QdrantVectorPoint]) -> None:
    """Write catalog or policy vector points to Qdrant.

    Args:
        points (list[QdrantVectorPoint]):
            Qdrant vector points to upsert into the configured collection.

    Returns:
        None:
            No value; points are persisted in Qdrant as a side effect.
    """
    if not points:
        return

    await ensure_qdrant_collection()
    async with _client() as client:
        response = await client.put(
            f"{_collection_url()}/points",
            params={"wait": "true"},
            json={
                "points": [
                    {
                        "id": _point_id(point),
                        "vector": point.vector,
                        "payload": _point_payload(point),
                    }
                    for point in points
                ]
            },
        )
        _raise_for_qdrant_error(response)


async def search_catalog_item_vectors(
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[SearchHit]:
    """Search Qdrant for catalog item source IDs similar to a query vector.

    Args:
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query_embedding (list[float]):
            Embedding vector produced from the user query.
        k (int):
            Maximum number of ranked candidates or results to return.

    Returns:
        list[SearchHit]:
            Ranked catalog search hits containing source IDs and scores.
    """
    hits = await _search_vectors(
        tenant_id=tenant_id,
        source_kind=QdrantSourceKind.CATALOG_ITEM,
        query_embedding=query_embedding,
        k=k,
    )
    return [
        SearchHit(
            item_id=source_id,
            score=score,
            source="semantic",
            rank=rank,
            kind="catalog",
        )
        for rank, (source_id, score) in enumerate(hits, start=1)
    ]


async def search_policy_chunk_vectors(
    tenant_id: int,
    query_embedding: list[float],
    k: int,
) -> list[tuple[int, float]]:
    """Search Qdrant for policy chunk source IDs similar to a query vector.

    Args:
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        query_embedding (list[float]):
            Embedding vector produced from the user query.
        k (int):
            Maximum number of ranked candidates or results to return.

    Returns:
        list[tuple[int, float]]:
            Policy chunk IDs paired with vector similarity scores.
    """
    return await _search_vectors(
        tenant_id=tenant_id,
        source_kind=QdrantSourceKind.POLICY_CHUNK,
        query_embedding=query_embedding,
        k=k,
    )


async def _search_vectors(
    *,
    tenant_id: int,
    source_kind: QdrantSourceKind,
    query_embedding: list[float],
    k: int,
) -> list[tuple[int, float]]:
    """Search vectors.

    Args:
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        source_kind (QdrantSourceKind):
            Qdrant source kind used to filter vector search results.
        query_embedding (list[float]):
            Embedding vector produced from the user query.
        k (int):
            Maximum number of ranked candidates or results to return.

    Returns:
        list[tuple[int, float]]:
            Ranked search results or source IDs matching the query constraints.
    """
    if k <= 0 or not query_embedding or not qdrant_enabled():
        return []

    async with _client() as client:
        response = await client.post(
            f"{_collection_url()}/points/search",
            json={
                "vector": query_embedding,
                "limit": k,
                "with_payload": True,
                "filter": {
                    "must": [
                        {"key": "tenant_id", "match": {"value": tenant_id}},
                        {"key": "source_kind", "match": {"value": source_kind.value}},
                        {
                            "key": "embedding_provider",
                            "match": {"value": settings.embedding_provider},
                        },
                        {
                            "key": "embedding_model_name",
                            "match": {"value": settings.embedding_model_name},
                        },
                        {
                            "key": "embedding_dimension",
                            "match": {"value": settings.embedding_dimension},
                        },
                    ]
                },
            },
        )
        _raise_for_qdrant_error(response)

    results = response.json().get("result", [])
    hits: list[tuple[int, float]] = []
    for result in results:
        payload = result.get("payload") or {}
        source_id = payload.get("source_id")
        score = result.get("score")
        if isinstance(source_id, int) and isinstance(score, int | float):
            hits.append((source_id, float(score)))
    return hits


async def _create_collection(client: httpx.AsyncClient) -> None:
    """Create collection.

    Args:
        client (httpx.AsyncClient):
            HTTP client used to call the Qdrant API.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    response = await client.put(
        _collection_url(),
        json={
            "vectors": {
                "size": settings.embedding_dimension,
                "distance": "Cosine",
            }
        },
    )
    _raise_for_qdrant_error(response)


def _validate_collection_config(payload: dict[str, Any]) -> None:
    """Validate collection config.

    Args:
        payload (dict[str, Any]):
            JSON-like payload read from an API request, test case, or trace.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    vectors = (
        payload.get("result", {})
        .get("config", {})
        .get("params", {})
        .get("vectors", {})
    )
    size = vectors.get("size") if isinstance(vectors, dict) else None
    if size is not None and int(size) != settings.embedding_dimension:
        raise RuntimeError(
            f"Qdrant collection {settings.qdrant_collection!r} has vector size {size}, "
            f"but {settings.embedding_dimension} is configured."
        )


def _point_payload(point: QdrantVectorPoint) -> dict[str, object]:
    """Handle point payload.

    Args:
        point (QdrantVectorPoint):
            Qdrant vector point being serialized or addressed.

    Returns:
        dict[str, object]:
            Value produced for the caller according to the function contract.
    """
    payload: dict[str, object] = {
        "tenant_id": point.tenant_id,
        "source_kind": point.source_kind.value,
        "source_id": point.source_id,
        "content_hash": point.content_hash,
        "embedding_provider": settings.embedding_provider,
        "embedding_model_name": settings.embedding_model_name,
        "embedding_dimension": settings.embedding_dimension,
    }
    if point.menu_version_id is not None:
        payload["menu_version_id"] = point.menu_version_id
    if point.policy_document_id is not None:
        payload["policy_document_id"] = point.policy_document_id
    return payload


def _point_id(point: QdrantVectorPoint) -> str:
    """Handle point ID.

    Args:
        point (QdrantVectorPoint):
            Qdrant vector point being serialized or addressed.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            (
                f"{settings.qdrant_collection}:"
                f"{point.tenant_id}:"
                f"{point.source_kind.value}:"
                f"{point.source_id}"
            ),
        )
    )


def _client() -> httpx.AsyncClient:
    """Handle client.

    Args:
        None.

    Returns:
        httpx.AsyncClient:
            Value produced for the caller according to the function contract.
    """
    return httpx.AsyncClient(
        timeout=30.0,
        headers=_headers(),
    )


def _headers() -> dict[str, str]:
    """Handle headers.

    Args:
        None.

    Returns:
        dict[str, str]:
            Value produced for the caller according to the function contract.
    """
    headers = {"Content-Type": "application/json"}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key
    return headers


def _collection_url() -> str:
    """Handle collection url.

    Args:
        None.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    if not settings.qdrant_url:
        raise RuntimeError("QDRANT_URL is required when VECTOR_PROVIDER=qdrant.")
    base_url = settings.qdrant_url.rstrip("/")
    return f"{base_url}/collections/{settings.qdrant_collection}"


def _raise_for_qdrant_error(response: httpx.Response) -> None:
    """Handle raise for Qdrant error.

    Args:
        response (httpx.Response):
            HTTP or chat response object being parsed or checked.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Qdrant request failed with HTTP {exc.response.status_code}."
        ) from exc
