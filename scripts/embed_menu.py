from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cafe_assistant.db.models import Ingredient, MenuItem
from cafe_assistant.db.session import async_session_maker
from cafe_assistant.gateway.model_gateway import EmbeddingProvider, get_embedding_provider
from cafe_assistant.retrieval.embeddings import build_menu_item_embedding_text


async def backfill_menu_embeddings(
    session: AsyncSession,
    provider: EmbeddingProvider | None = None,
    tenant_id: int | None = None,
) -> int:
    embedding_provider = provider or get_embedding_provider()
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


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill menu item embeddings.")
    parser.add_argument("--tenant-id", type=int, default=None)
    args = parser.parse_args()

    async with async_session_maker() as session:
        updated_count = await backfill_menu_embeddings(session, tenant_id=args.tenant_id)
    print(f"Embedded {updated_count} menu items.")


if __name__ == "__main__":
    asyncio.run(main())
