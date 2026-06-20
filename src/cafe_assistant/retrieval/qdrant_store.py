"""Qdrant vector-index adapter for catalog-item and policy-chunk pointers.

Qdrant is used only as a similarity index. It stores vectors plus routing metadata
such as tenant ID, source kind, source row ID, embedding provider, model name, and
dimension. It does not own menu text, policy text, allergen facts, dietary facts,
or availability. Retrieval must resolve returned IDs back to Postgres/Neon before
anything can be shown to the agent or user.

The adapter normalizes network, HTTP, and malformed-response failures into
`QdrantVectorStoreError` so callers can fall back to SQL/pgvector paths without
leaking raw transport exceptions into user-facing flows.
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
    """Source families stored in the shared Qdrant collection."""

    CATALOG_ITEM = "catalog_item"
    POLICY_CHUNK = "policy_chunk"


@dataclass(frozen=True, slots=True)
class QdrantVectorPoint:
    """Vector and routing metadata for one Qdrant point upsert.

    Catalog item points use `source_kind=CATALOG_ITEM` and `source_id` as the
    catalog variant ID. Policy points use `source_kind=POLICY_CHUNK` and
    `source_id` as the policy chunk ID. Optional version/document IDs are stored
    as payload metadata for replay and debugging, not as authoritative content.
    """

    tenant_id: int
    source_kind: QdrantSourceKind
    source_id: int
    vector: list[float]
    content_hash: str
    menu_version_id: int | None = None
    policy_document_id: int | None = None


class QdrantVectorStoreError(RuntimeError):
    """Raised when Qdrant cannot complete an index operation safely."""


def qdrant_enabled() -> bool:
    """Return whether runtime settings select Qdrant as the vector index.

    Args:
        None.

    Returns:
        bool:
            True when `VECTOR_PROVIDER=qdrant` and a Qdrant URL is configured;
            otherwise false so retrieval can use the SQL/pgvector fallback path.
    """
    return settings.vector_provider == "qdrant" and bool(settings.qdrant_url)


async def ensure_qdrant_collection() -> None:
    """Create the configured Qdrant collection or verify its vector size.

    Args:
        None.

    Returns:
        None:
            The collection exists with the configured vector dimension. A
            `QdrantVectorStoreError` or `RuntimeError` is raised when Qdrant
            rejects the request or the existing collection shape is incompatible.
    """
    async with _client() as client:
        response = await client.get(_collection_url())
        if response.status_code == 404:
            await _create_collection(client)
            return
        _raise_for_qdrant_error(response)
        _validate_collection_config(response.json())


async def upsert_qdrant_points(points: list[QdrantVectorPoint]) -> None:
    """Upsert catalog or policy vectors into the configured Qdrant collection.

    Args:
        points (list[QdrantVectorPoint]):
            Vector points produced by the embedding backfill script. Each point
            includes tenant, source-kind, source-ID, and embedding-version payload
            fields required for tenant-safe retrieval.

    Returns:
        None:
            Points are persisted in Qdrant. Empty input is a no-op.
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
    """Search Qdrant for catalog variant IDs similar to a query vector.

    Args:
        tenant_id (int):
            Tenant scope required in the Qdrant payload filter.
        query_embedding (list[float]):
            Query vector produced by the configured embedding provider/model.
        k (int):
            Maximum number of catalog vector hits to request from Qdrant.

    Returns:
        list[SearchHit]:
            Ranked semantic hits whose `item_id` values are catalog variant IDs.
            Callers must reload these IDs from SQL before safety filtering or
            response composition.
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
    """Search Qdrant for policy chunk IDs similar to a query vector.

    Args:
        tenant_id (int):
            Tenant scope required in the Qdrant payload filter.
        query_embedding (list[float]):
            Query vector produced by the configured embedding provider/model.
        k (int):
            Maximum number of policy vector hits to request from Qdrant.

    Returns:
        list[tuple[int, float]]:
            Policy chunk IDs paired with Qdrant similarity scores. Callers must
            reload chunk text from SQL before adding it to model context.
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
    """Run a tenant-scoped Qdrant vector search for one source family.

    Args:
        tenant_id (int):
            Tenant ID applied as a mandatory Qdrant payload filter.
        source_kind (QdrantSourceKind):
            Source family filter that prevents menu and policy vectors from being
            mixed in one search result set.
        query_embedding (list[float]):
            Query vector with the configured embedding dimension.
        k (int):
            Maximum number of matching points to return.

    Returns:
        list[tuple[int, float]]:
            Source row IDs and similarity scores from valid Qdrant result rows.
            Malformed rows are ignored because SQL reload will be the authority.

    Raises:
        QdrantVectorStoreError:
            Raised for Qdrant HTTP/client failures or malformed top-level JSON so
            callers can record fallback metrics and continue safely.
    """
    if k <= 0 or not query_embedding or not qdrant_enabled():
        return []

    try:
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
        response_payload = response.json()
        if not isinstance(response_payload, dict):
            raise ValueError("Qdrant search response root must be an object.")
        results = response_payload.get("result", [])
    except httpx.HTTPError as exc:
        raise QdrantVectorStoreError("Qdrant search request failed.") from exc
    except ValueError as exc:
        raise QdrantVectorStoreError("Qdrant search response was not valid JSON.") from exc

    hits: list[tuple[int, float]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        payload = result.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        source_id = payload.get("source_id")
        score = result.get("score")
        if isinstance(source_id, int) and isinstance(score, int | float):
            hits.append((source_id, float(score)))
    return hits


async def _create_collection(client: httpx.AsyncClient) -> None:
    """Create the Qdrant collection with the configured vector dimension.

    Args:
        client (httpx.AsyncClient):
            Authenticated HTTP client used to call the Qdrant API.

    Returns:
        None:
            The collection is created or a `QdrantVectorStoreError` is raised for
            an unsuccessful Qdrant response.
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
    """Validate that an existing Qdrant collection matches embedding settings.

    Args:
        payload (dict[str, Any]):
            JSON response body returned by Qdrant's collection metadata endpoint.

    Returns:
        None:
            Validation succeeds when Qdrant reports no size or the size matches
            `settings.embedding_dimension`.

    Raises:
        RuntimeError:
            Raised when the existing collection dimension does not match the
            configured embedding dimension.
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
    """Build the Qdrant payload stored beside one vector.

    Args:
        point (QdrantVectorPoint):
            Vector point containing tenant, source, content-hash, and optional
            version metadata.

    Returns:
        dict[str, object]:
            Payload used for filtering and incident replay. It intentionally
            contains IDs and hashes, not raw menu or policy text.
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
    """Create a stable UUID for one tenant/source row in one collection.

    Args:
        point (QdrantVectorPoint):
            Vector point whose tenant, source kind, and source row ID identify the
            logical Qdrant record.

    Returns:
        str:
            Deterministic UUID string. Re-running embedding backfill for the same
            source updates the same Qdrant point instead of creating duplicates.
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
    """Create the short-lived HTTP client used for Qdrant API calls.

    Args:
        None.

    Returns:
        httpx.AsyncClient:
            Client configured with Qdrant headers and the retrieval timeout from
            settings so vector outages fail fast and callers can fall back.
    """
    return httpx.AsyncClient(
        timeout=settings.qdrant_timeout_seconds,
        headers=_headers(),
    )


def _headers() -> dict[str, str]:
    """Build HTTP headers for Qdrant requests.

    Args:
        None.

    Returns:
        dict[str, str]:
            Headers containing JSON content type and the optional Qdrant API key.
            The API key is passed as a header and is never logged by this module.
    """
    headers = {"Content-Type": "application/json"}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key
    return headers


def _collection_url() -> str:
    """Build the configured Qdrant collection URL.

    Args:
        None.

    Returns:
        str:
            Fully qualified Qdrant collection endpoint.

    Raises:
        RuntimeError:
            Raised when Qdrant was selected but no Qdrant URL was configured.
    """
    if not settings.qdrant_url:
        raise RuntimeError("QDRANT_URL is required when VECTOR_PROVIDER=qdrant.")
    base_url = settings.qdrant_url.rstrip("/")
    return f"{base_url}/collections/{settings.qdrant_collection}"


def _raise_for_qdrant_error(response: httpx.Response) -> None:
    """Convert unsuccessful Qdrant HTTP responses into adapter errors.

    Args:
        response (httpx.Response):
            HTTP response returned by Qdrant.

    Returns:
        None:
            The response was successful.

    Raises:
        QdrantVectorStoreError:
            Raised when Qdrant responds with a non-2xx HTTP status.
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise QdrantVectorStoreError(
            f"Qdrant request failed with HTTP {exc.response.status_code}."
        ) from exc