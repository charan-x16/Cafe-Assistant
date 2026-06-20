"""Tests for legacy embeddings.
Exercises expected behavior with deterministic fixtures and mocked providers where needed.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cafe_assistant.db.models import Ingredient, MenuItem
from cafe_assistant.gateway.model_gateway import EmbeddingProvider
from cafe_assistant.retrieval.embeddings import build_menu_item_embedding_text


async def backfill_menu_embeddings(
    session: AsyncSession,
    provider: EmbeddingProvider | None = None,
    tenant_id: int | None = None,
) -> int:
    """Backfill menu embeddings.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        provider (EmbeddingProvider | None):
            Optional embedding provider override used by tests or scripts.
        tenant_id (int | None):
            Tenant identifier used to scope database and vector-store operations.

    Returns:
        int:
            Value produced for the caller according to the function contract.
    """
    if provider is None:
        raise ValueError('Test fixture requires an explicit embedding provider.')
    embedding_provider = provider
    statement = (
        select(MenuItem)
        .where(MenuItem.is_available.is_(True))
        .options(
            selectinload(MenuItem.dietary_tags),
            selectinload(MenuItem.ingredients).selectinload(Ingredient.allergens),
        )
        .order_by(MenuItem.id)
    )
    if tenant_id is not None:
        statement = statement.where(MenuItem.tenant_id == tenant_id)

    result = await session.scalars(statement)
    items = list(result.unique())
    if not items:
        return 0

    embedding_texts = [build_menu_item_embedding_text(item) for item in items]
    embeddings = embedding_provider.embed(embedding_texts)
    if len(embeddings) != len(items):
        raise ValueError("Embedding provider returned a different number of vectors than inputs.")

    for item, embedding in zip(items, embeddings, strict=True):
        item.embedding = embedding

    await session.commit()
    return len(items)

