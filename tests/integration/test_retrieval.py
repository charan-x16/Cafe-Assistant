from __future__ import annotations

import math
import re
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cafe_assistant.db.base import Base
from cafe_assistant.db.models import Tenant
from cafe_assistant.domain.dietary import AllergenCode, CustomerRestrictions, DietaryMode
from cafe_assistant.gateway.model_gateway import EmbeddingProvider
from cafe_assistant.retrieval.hybrid import search_menu
from scripts.embed_menu import backfill_menu_embeddings
from scripts.seed_menu import TENANT_NAME, seed_database


class FakeEmbeddingProvider:
    dimensions = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        vector = [
            self._feature(tokens, {"coffee", "espresso", "cappuccino", "latte", "mocha"}),
            self._feature(tokens, {"tea", "chai", "matcha", "earl"}),
            self._feature(tokens, {"almond", "almnd", "nut"}),
            self._feature(tokens, {"cookie", "peanut", "butter"}),
            self._feature(tokens, {"sandwich", "panini", "toast"}),
            self._feature(tokens, {"chocolate", "mocha"}),
            self._feature(tokens, {"gluten", "bread", "sourdough", "dough"}),
            self._feature(tokens, {"vegan", "vegetarian", "gluten_free"}),
        ]
        magnitude = math.sqrt(sum(component * component for component in vector))
        if magnitude == 0:
            return vector
        return [component / magnitude for component in vector]

    def _feature(self, tokens: set[str], vocabulary: set[str]) -> float:
        return float(len(tokens & vocabulary))


@pytest.fixture
async def seeded_session() -> AsyncIterator[tuple[AsyncSession, int, EmbeddingProvider]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        provider = FakeEmbeddingProvider()
        await seed_database(session)
        tenant_id = await session.scalar(select(Tenant.id).where(Tenant.name == TENANT_NAME))
        assert tenant_id is not None
        await backfill_menu_embeddings(session, provider=provider, tenant_id=tenant_id)
        yield session, tenant_id, provider

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def test_exact_query_returns_matching_item(
    seeded_session: tuple[AsyncSession, int, EmbeddingProvider],
) -> None:
    session, tenant_id, provider = seeded_session

    results = await search_menu(
        session,
        tenant_id,
        "cappuccino",
        CustomerRestrictions(avoid_allergens=set(), modes=set()),
        embedding_provider=provider,
    )

    assert results
    assert results[0].name == "Cappuccino"


async def test_fuzzy_query_returns_plausible_items(
    seeded_session: tuple[AsyncSession, int, EmbeddingProvider],
) -> None:
    session, tenant_id, provider = seeded_session

    results = await search_menu(
        session,
        tenant_id,
        "almnd latte",
        CustomerRestrictions(avoid_allergens=set(), modes=set()),
        embedding_provider=provider,
    )

    result_names = [item.name for item in results[:3]]
    assert "Matcha Almond Latte" in result_names


async def test_unsafe_semantic_match_is_filtered_out(
    seeded_session: tuple[AsyncSession, int, EmbeddingProvider],
) -> None:
    session, tenant_id, provider = seeded_session
    restrictions = CustomerRestrictions(
        avoid_allergens={AllergenCode.PEANUT},
        modes=set(),
    )

    results = await search_menu(
        session,
        tenant_id,
        "peanut butter cookie",
        restrictions,
        embedding_provider=provider,
    )

    assert results
    assert "Peanut Butter Cookie" not in {item.name for item in results}
    assert all(AllergenCode.PEANUT not in item.allergen_codes for item in results)
    assert all(item.allergen_data_complete for item in results)


async def test_search_menu_returns_only_post_filter_safe_items(
    seeded_session: tuple[AsyncSession, int, EmbeddingProvider],
) -> None:
    session, tenant_id, provider = seeded_session
    restrictions = CustomerRestrictions(
        avoid_allergens={AllergenCode.DAIRY, AllergenCode.GLUTEN},
        modes={DietaryMode.VEGAN},
    )

    results = await search_menu(
        session,
        tenant_id,
        "latte sandwich pastry",
        restrictions,
        k=20,
        embedding_provider=provider,
    )

    assert results
    for item in results:
        assert item.allergen_data_complete is True
        assert AllergenCode.DAIRY not in item.allergen_codes
        assert AllergenCode.GLUTEN not in item.allergen_codes
        assert DietaryMode.VEGAN in item.dietary_tags


async def test_low_sugar_preference_reorders_safe_results(
    seeded_session: tuple[AsyncSession, int, EmbeddingProvider],
) -> None:
    session, tenant_id, provider = seeded_session
    restrictions = CustomerRestrictions(
        avoid_allergens=set(),
        modes=set(),
        prefer_low_sugar=True,
    )

    results = await search_menu(
        session,
        tenant_id,
        "latte",
        restrictions,
        k=5,
        embedding_provider=provider,
    )

    known_sugars = [item.sugar_grams for item in results if item.sugar_grams is not None]
    assert known_sugars == sorted(known_sugars)
