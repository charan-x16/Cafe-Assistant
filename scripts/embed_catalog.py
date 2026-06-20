"""Backfill production catalog and policy embeddings.
Reads published catalog variants and policy chunks from SQL, builds embedding text, and writes
vectors to Qdrant or the pgvector fallback tables.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cafe_assistant.config import settings
from cafe_assistant.db.models import (
    CatalogItem,
    CatalogItemAllergenAssertion,
    CatalogItemDietaryAssertion,
    CatalogItemEmbedding,
    CatalogItemVariant,
    Menu,
    MenuVersion,
    PolicyChunk,
    PolicyChunkEmbedding,
    PolicyDocument,
)
from cafe_assistant.db.session import async_session_maker
from cafe_assistant.gateway.model_gateway import EmbeddingProvider, get_embedding_provider
from cafe_assistant.retrieval.embeddings import (
    build_catalog_variant_embedding_text,
    build_policy_chunk_embedding_text,
)
from cafe_assistant.retrieval.qdrant_store import (
    QdrantSourceKind,
    QdrantVectorPoint,
    upsert_qdrant_points,
)


async def backfill_catalog_embeddings(
    session: AsyncSession,
    provider: EmbeddingProvider | None = None,
    tenant_id: int | None = None,
) -> int:
    """Create embeddings for published catalog variants and policy chunks.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        provider (EmbeddingProvider | None):
            Optional embedding provider override used by tests or scripts.
        tenant_id (int | None):
            Tenant identifier used to scope database and vector-store operations.

    Returns:
        int:
            Count of catalog variants and policy chunks embedded.
    """
    embedding_provider = provider or get_embedding_provider()
    variants = await _load_published_variants(session, tenant_id)
    chunks = await _load_policy_chunks(session, tenant_id)
    records: list[tuple[str, object, str]] = [
        ("catalog_item", variant, build_catalog_variant_embedding_text(variant))
        for variant in variants
    ] + [
        ("policy_chunk", chunk, build_policy_chunk_embedding_text(chunk))
        for chunk in chunks
    ]
    if not records:
        return 0

    texts = [record[2] for record in records]
    embeddings = embedding_provider.embed(texts)
    if len(embeddings) != len(records):
        raise ValueError("Embedding provider returned a different number of vectors than inputs.")

    qdrant_points: list[QdrantVectorPoint] = []
    use_qdrant = _use_qdrant_vector_store(provider)
    updated = 0
    for (record_type, source, embedded_text), embedding in zip(records, embeddings, strict=True):
        _validate_embedding(embedding)
        if use_qdrant:
            qdrant_points.append(_qdrant_point(record_type, source, embedded_text, embedding))
        elif record_type == "catalog_item":
            await _upsert_catalog_item_embedding(session, source, embedded_text, embedding)
        else:
            await _upsert_policy_chunk_embedding(session, source, embedded_text, embedding)
        updated += 1

    if qdrant_points:
        await upsert_qdrant_points(qdrant_points)

    await session.commit()
    return updated


async def _load_published_variants(
    session: AsyncSession,
    tenant_id: int | None,
) -> list[CatalogItemVariant]:
    """Load published variants.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int | None):
            Tenant identifier used to scope database and vector-store operations.

    Returns:
        list[CatalogItemVariant]:
            Loaded records or projected domain values matching the requested scope.
    """
    statement = (
        select(CatalogItemVariant)
        .join(CatalogItem)
        .join(MenuVersion)
        .join(Menu)
        .where(MenuVersion.status == "published")
        .where(CatalogItem.is_available.is_(True))
        .where(CatalogItemVariant.is_available.is_(True))
        .options(
            selectinload(CatalogItemVariant.catalog_item).selectinload(CatalogItem.category),
            selectinload(CatalogItemVariant.catalog_item).selectinload(CatalogItem.ingredients),
            selectinload(CatalogItemVariant.catalog_item)
            .selectinload(CatalogItem.menu_version)
            .selectinload(MenuVersion.menu),
            selectinload(CatalogItemVariant.catalog_item)
            .selectinload(CatalogItem.allergen_assertions)
            .selectinload(CatalogItemAllergenAssertion.allergen),
            selectinload(CatalogItemVariant.catalog_item)
            .selectinload(CatalogItem.dietary_assertions)
            .selectinload(CatalogItemDietaryAssertion.dietary_tag),
        )
        .order_by(CatalogItem.sort_order, CatalogItemVariant.sort_order, CatalogItemVariant.id)
    )
    if tenant_id is not None:
        statement = statement.where(Menu.tenant_id == tenant_id)
    return list((await session.scalars(statement)).unique())


async def _load_policy_chunks(
    session: AsyncSession,
    tenant_id: int | None,
) -> list[PolicyChunk]:
    """Load policy chunks.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int | None):
            Tenant identifier used to scope database and vector-store operations.

    Returns:
        list[PolicyChunk]:
            Loaded records or projected domain values matching the requested scope.
    """
    statement = (
        select(PolicyChunk)
        .join(PolicyDocument)
        .options(selectinload(PolicyChunk.policy_document))
        .order_by(PolicyDocument.id, PolicyChunk.chunk_index)
    )
    if tenant_id is not None:
        statement = statement.where(PolicyDocument.tenant_id == tenant_id)
    return list((await session.scalars(statement)).unique())


async def _upsert_catalog_item_embedding(
    session: AsyncSession,
    source: object,
    embedded_text: str,
    embedding: list[float],
) -> None:
    """Insert or update catalog item embedding.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        source (object):
            Database source row converted into an embedding record.
        embedded_text (str):
            Exact text used to create the embedding and content hash.
        embedding (list[float]):
            Embedding vector to validate, store, or compare.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    if not isinstance(source, CatalogItemVariant):
        raise TypeError("Expected CatalogItemVariant for catalog embedding.")
    embedded_text_hash = _hash_text(embedded_text)
    row = await session.scalar(
        select(CatalogItemEmbedding)
        .where(CatalogItemEmbedding.variant_id == source.id)
        .where(CatalogItemEmbedding.provider == settings.embedding_provider)
        .where(CatalogItemEmbedding.model_name == _embedding_model_name())
        .where(CatalogItemEmbedding.embedded_text_hash == embedded_text_hash)
    )
    if row is None:
        row = CatalogItemEmbedding(
            variant=source,
            provider=settings.embedding_provider,
            model_name=_embedding_model_name(),
            dimensions=settings.embedding_dimension,
            embedded_text_hash=embedded_text_hash,
            embedded_text=embedded_text,
            embedding=embedding,
        )
        session.add(row)
        return
    row.embedding = embedding
    row.embedded_text = embedded_text


async def _upsert_policy_chunk_embedding(
    session: AsyncSession,
    source: object,
    embedded_text: str,
    embedding: list[float],
) -> None:
    """Insert or update policy chunk embedding.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        source (object):
            Database source row converted into an embedding record.
        embedded_text (str):
            Exact text used to create the embedding and content hash.
        embedding (list[float]):
            Embedding vector to validate, store, or compare.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    if not isinstance(source, PolicyChunk):
        raise TypeError("Expected PolicyChunk for policy embedding.")
    embedded_text_hash = _hash_text(embedded_text)
    row = await session.scalar(
        select(PolicyChunkEmbedding)
        .where(PolicyChunkEmbedding.policy_chunk_id == source.id)
        .where(PolicyChunkEmbedding.provider == settings.embedding_provider)
        .where(PolicyChunkEmbedding.model_name == _embedding_model_name())
        .where(PolicyChunkEmbedding.embedded_text_hash == embedded_text_hash)
    )
    if row is None:
        row = PolicyChunkEmbedding(
            policy_chunk=source,
            provider=settings.embedding_provider,
            model_name=_embedding_model_name(),
            dimensions=settings.embedding_dimension,
            embedded_text_hash=embedded_text_hash,
            embedded_text=embedded_text,
            embedding=embedding,
        )
        session.add(row)
        return
    row.embedding = embedding
    row.embedded_text = embedded_text


def _validate_embedding(embedding: list[float]) -> None:
    """Validate embedding.

    Args:
        embedding (list[float]):
            Embedding vector to validate, store, or compare.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    if len(embedding) != settings.embedding_dimension:
        raise ValueError(
            f"Expected embedding dimension {settings.embedding_dimension}, got {len(embedding)}."
        )


def _use_qdrant_vector_store(provider: EmbeddingProvider | None) -> bool:
    """Handle use Qdrant vector store.

    Args:
        provider (EmbeddingProvider | None):
            Optional embedding provider override used by tests or scripts.

    Returns:
        bool:
            Value produced for the caller according to the function contract.
    """
    return settings.vector_provider == "qdrant" and provider is None


def _qdrant_point(
    record_type: str,
    source: object,
    embedded_text: str,
    embedding: list[float],
) -> QdrantVectorPoint:
    """Handle Qdrant point.

    Args:
        record_type (str):
            Kind of source record represented by the embedding point.
        source (object):
            Database source row converted into an embedding record.
        embedded_text (str):
            Exact text used to create the embedding and content hash.
        embedding (list[float]):
            Embedding vector to validate, store, or compare.

    Returns:
        QdrantVectorPoint:
            Value produced for the caller according to the function contract.
    """
    embedded_text_hash = _hash_text(embedded_text)
    if record_type == "catalog_item" and isinstance(source, CatalogItemVariant):
        catalog_item = source.catalog_item
        menu_version = catalog_item.menu_version
        return QdrantVectorPoint(
            tenant_id=menu_version.menu.tenant_id,
            source_kind=QdrantSourceKind.CATALOG_ITEM,
            source_id=source.id,
            vector=embedding,
            content_hash=embedded_text_hash,
            menu_version_id=menu_version.id,
        )
    if record_type == "policy_chunk" and isinstance(source, PolicyChunk):
        policy_document = source.policy_document
        return QdrantVectorPoint(
            tenant_id=policy_document.tenant_id,
            source_kind=QdrantSourceKind.POLICY_CHUNK,
            source_id=source.id,
            vector=embedding,
            content_hash=embedded_text_hash,
            policy_document_id=policy_document.id,
        )
    raise TypeError(f"Unsupported Qdrant source record: {record_type}")


def _embedding_model_name() -> str:
    """Handle embedding model name.

    Args:
        None.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    return (
        f"{settings.embedding_provider}:{settings.embedding_model_name}"
        f"_dim_{settings.embedding_dimension}_v1"
    )


def _hash_text(text_value: str) -> str:
    """Hash text.

    Args:
        text_value (str):
            Raw text being parsed, cleaned, hashed, or tokenized.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    return hashlib.sha256(text_value.encode("utf-8")).hexdigest()


async def main_async(tenant_id: int | None) -> int:
    """Run the asynchronous command workflow for this module.

    Args:
        tenant_id (int | None):
            Tenant identifier used to scope database and vector-store operations.

    Returns:
        int:
            Value produced for the caller according to the function contract.
    """
    async with async_session_maker() as session:
        return await backfill_catalog_embeddings(session, tenant_id=tenant_id)


def main() -> None:
    """Run the command-line interface for this module.

    Args:
        None.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    parser = argparse.ArgumentParser(
        description="Backfill production catalog and policy chunk embeddings."
    )
    parser.add_argument("--tenant-id", type=int, default=None)
    args = parser.parse_args()

    updated_count = asyncio.run(main_async(args.tenant_id))
    print(f"Embedded {updated_count} catalog records.")


if __name__ == "__main__":
    main()
