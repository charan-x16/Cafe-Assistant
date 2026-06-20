"""Tests for models.
Exercises expected behavior with deterministic fixtures and mocked providers where needed.
"""

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cafe_assistant.db.base import Base
from cafe_assistant.db.models import Allergen, Ingredient, MenuItem, Tenant, ingredient_allergens
from cafe_assistant.main import app
from tests.fixtures.legacy_menu import seed_database


def test_schema_metadata_loads() -> None:
    """Verify that schema metadata loads.

    Args:
        None.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    expected_tables = {
        "tenants",
        "locations",
        "menu_items",
        "ingredients",
        "item_ingredients",
        "allergens",
        "ingredient_allergens",
        "dietary_tags",
        "item_dietary_tags",
        "source_documents",
        "menu_import_batches",
        "menus",
        "menu_versions",
        "menu_categories",
        "catalog_items",
        "catalog_item_variants",
        "catalog_item_allergen_assertions",
        "catalog_item_dietary_assertions",
        "modifier_groups",
        "modifier_options",
        "policy_documents",
        "policy_chunks",
    }

    assert expected_tables <= set(Base.metadata.tables)
    assert Base.metadata.tables["menu_items"].c.allergen_data_complete.default is not None


def test_health_endpoint() -> None:
    """Verify that health endpoint.

    Args:
        None.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_legacy_menu_fixture_inserts_data() -> None:
    """Verify that legacy menu fixture inserts data.

    Args:
        None.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    try:
        async with session_factory() as session:
            inserted = await seed_database(session)

            tenant_count = await session.scalar(select(func.count()).select_from(Tenant))
            item_count = await session.scalar(select(func.count()).select_from(MenuItem))
            ingredient_count = await session.scalar(select(func.count()).select_from(Ingredient))
            allergen_count = await session.scalar(select(func.count()).select_from(Allergen))
            mapped_allergen_count = await session.scalar(
                select(func.count()).select_from(ingredient_allergens)
            )
            incomplete_item_count = await session.scalar(
                select(func.count())
                .select_from(MenuItem)
                .where(MenuItem.allergen_data_complete.is_(False))
            )

        assert inserted is True
        assert tenant_count == 1
        assert item_count == 16
        assert ingredient_count >= 20
        assert allergen_count == 6
        assert mapped_allergen_count > 0
        assert incomplete_item_count >= 2
    finally:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await engine.dispose()
