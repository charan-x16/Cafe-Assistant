"""Unit tests for the Qdrant vector adapter used by retrieval.

These tests keep Qdrant mocked at the HTTP-client boundary so they can verify
request filters, source separation, and error normalization without calling a
real Qdrant Cloud collection.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from cafe_assistant.retrieval import qdrant_store
from cafe_assistant.retrieval.qdrant_store import (
    QdrantSourceKind,
    QdrantVectorStoreError,
    search_catalog_item_vectors,
    search_policy_chunk_vectors,
)


class FakeQdrantResponse:
    """Small response double that mimics the Qdrant response methods we consume."""

    status_code = 200

    def __init__(self, payload: dict[str, Any] | list[Any]) -> None:
        """Store the JSON payload returned by the fake Qdrant request.

        Args:
            payload (dict[str, Any] | list[Any]):
                JSON-like value returned by the fake response's `json()` method.

        Returns:
            None:
                The response double is initialized for later assertions.
        """
        self.payload = payload

    def json(self) -> dict[str, Any] | list[Any]:
        """Return the configured fake JSON payload.

        Args:
            None.

        Returns:
            dict[str, Any] | list[Any]:
                Payload supplied when this response was constructed.
        """
        return self.payload

    def raise_for_status(self) -> None:
        """Mirror a successful HTTP response with no raised exception.

        Args:
            None.

        Returns:
            None:
                No exception is raised for this fake success response.
        """
        return None


class FakeQdrantClient:
    """Async context-manager double for the Qdrant HTTP client."""

    def __init__(
        self,
        response: FakeQdrantResponse | None = None,
        error: httpx.HTTPError | None = None,
    ) -> None:
        """Configure the fake client response or network-style exception.

        Args:
            response (FakeQdrantResponse | None):
                Response returned by `post()` when no error is configured.
            error (httpx.HTTPError | None):
                HTTP/client exception raised by `post()` to simulate Qdrant failure.

        Returns:
            None:
                The fake client is initialized with an empty request log.
        """
        self.response = response or FakeQdrantResponse({"result": []})
        self.error = error
        self.requests: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> FakeQdrantClient:
        """Enter the async context manager used by the production adapter.

        Args:
            None.

        Returns:
            FakeQdrantClient:
                The fake client itself so calls can be recorded.
        """
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the async context manager without suppressing exceptions.

        Args:
            *exc_info (object):
                Exception details passed by the async context manager protocol.

        Returns:
            None:
                No exception is suppressed by this fake client.
        """
        return None

    async def post(self, url: str, json: dict[str, Any]) -> FakeQdrantResponse:
        """Record the Qdrant search request and return or raise the configured result.

        Args:
            url (str):
                Qdrant endpoint URL requested by the adapter.
            json (dict[str, Any]):
                Request body sent to the Qdrant search endpoint.

        Returns:
            FakeQdrantResponse:
                Configured fake response when no error is configured.
        """
        self.requests.append((url, json))
        if self.error is not None:
            raise self.error
        return self.response


def _enable_qdrant(monkeypatch: pytest.MonkeyPatch, client: FakeQdrantClient) -> None:
    """Point the Qdrant adapter at a fake enabled collection and client.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Pytest monkeypatch helper used to isolate global settings.
        client (FakeQdrantClient):
            Fake client returned by the adapter's private client factory.

    Returns:
        None:
            Qdrant settings and client factory are patched for the test.
    """
    monkeypatch.setattr(qdrant_store.settings, "vector_provider", "qdrant")
    monkeypatch.setattr(qdrant_store.settings, "qdrant_url", "https://qdrant.test")
    monkeypatch.setattr(qdrant_store.settings, "embedding_provider", "sentence_transformer")
    monkeypatch.setattr(qdrant_store.settings, "embedding_model_name", "BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(qdrant_store.settings, "embedding_dimension", 384)
    monkeypatch.setattr(qdrant_store, "_client", lambda: client)


def _filter_value(request_body: dict[str, Any], key: str) -> object:
    """Read one equality filter value from a Qdrant search request body.

    Args:
        request_body (dict[str, Any]):
            JSON body sent to Qdrant's points search endpoint.
        key (str):
            Payload filter key to inspect.

    Returns:
        object:
            Matched value configured for the requested payload key.
    """
    for condition in request_body["filter"]["must"]:
        if condition["key"] == key:
            return condition["match"]["value"]
    raise AssertionError(f"Missing Qdrant filter for {key}.")


async def test_catalog_vector_search_sends_tenant_model_and_source_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify catalog Qdrant search is tenant/model scoped and returns catalog hits.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Pytest monkeypatch helper used to replace the Qdrant client.

    Returns:
        None:
            Assertions fail if the adapter omits required Qdrant filters.
    """
    client = FakeQdrantClient(
        FakeQdrantResponse(
            {
                "result": [
                    {"payload": {"source_id": 101}, "score": 0.91},
                    {"payload": {"source_id": "bad"}, "score": 0.99},
                ]
            }
        )
    )
    _enable_qdrant(monkeypatch, client)

    hits = await search_catalog_item_vectors(tenant_id=7, query_embedding=[0.1, 0.2], k=3)

    assert [hit.item_id for hit in hits] == [101]
    assert hits[0].kind == "catalog"
    assert hits[0].source == "semantic"
    assert len(client.requests) == 1
    _url, body = client.requests[0]
    assert body["vector"] == [0.1, 0.2]
    assert body["limit"] == 3
    assert _filter_value(body, "tenant_id") == 7
    assert _filter_value(body, "source_kind") == QdrantSourceKind.CATALOG_ITEM.value
    assert _filter_value(body, "embedding_provider") == "sentence_transformer"
    assert _filter_value(body, "embedding_model_name") == "BAAI/bge-small-en-v1.5"
    assert _filter_value(body, "embedding_dimension") == 384


async def test_policy_vector_search_uses_policy_source_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify policy retrieval cannot accidentally search catalog-item vectors.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Pytest monkeypatch helper used to replace the Qdrant client.

    Returns:
        None:
            Assertions fail if policy search omits source-kind separation.
    """
    client = FakeQdrantClient(
        FakeQdrantResponse({"result": [{"payload": {"source_id": 55}, "score": 0.77}]})
    )
    _enable_qdrant(monkeypatch, client)

    hits = await search_policy_chunk_vectors(tenant_id=3, query_embedding=[0.4], k=2)

    assert hits == [(55, 0.77)]
    _url, body = client.requests[0]
    assert _filter_value(body, "tenant_id") == 3
    assert _filter_value(body, "source_kind") == QdrantSourceKind.POLICY_CHUNK.value


async def test_qdrant_network_error_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify network/client failures become the adapter's safe retrieval error.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Pytest monkeypatch helper used to replace the Qdrant client.

    Returns:
        None:
            Assertions fail if a raw HTTPX exception escapes the adapter.
    """
    client = FakeQdrantClient(error=httpx.ConnectError("qdrant unavailable"))
    _enable_qdrant(monkeypatch, client)

    with pytest.raises(QdrantVectorStoreError):
        await search_catalog_item_vectors(tenant_id=1, query_embedding=[0.1], k=1)


async def test_qdrant_non_object_response_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify malformed Qdrant JSON is treated as a vector-store failure.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Pytest monkeypatch helper used to replace the Qdrant client.

    Returns:
        None:
            Assertions fail if malformed response data escapes as an arbitrary exception.
    """
    client = FakeQdrantClient(FakeQdrantResponse([]))
    _enable_qdrant(monkeypatch, client)

    with pytest.raises(QdrantVectorStoreError):
        await search_policy_chunk_vectors(tenant_id=1, query_embedding=[0.1], k=1)