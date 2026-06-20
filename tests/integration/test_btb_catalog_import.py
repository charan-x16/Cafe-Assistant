"""Tests for BTB catalog import.
Exercises expected behavior with deterministic fixtures and mocked providers where needed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cafe_assistant.db.base import Base
from cafe_assistant.db.models import (
    CatalogItem,
    CatalogItemAllergenAssertion,
    CatalogItemVariant,
    MenuImportBatch,
    MenuVersion,
    ModifierOption,
    PolicyChunk,
    SourceDocument,
    Tenant,
)
from cafe_assistant.db.repositories.menu_repo import load_published_catalog_item_views_for_tenant
from cafe_assistant.domain.dietary import (
    EXCLUDED_INCOMPLETE_DATA,
    AllergenCode,
    CustomerRestrictions,
    filter_safe_items,
)
from cafe_assistant.ingestion.btb_markdown import (
    BTB_TENANT_NAME,
    import_btb_documents,
)


@pytest.fixture
async def catalog_session() -> AsyncIterator[AsyncSession]:
    """Handle catalog session.

    Args:
        None.

    Returns:
        AsyncIterator[AsyncSession]:
            Streamed values yielded to the caller as they become available.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        yield session

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def test_btb_documents_import_into_published_catalog(
    catalog_session: AsyncSession,
) -> None:
    """Verify that BTB documents import into published catalog.

    Args:
        catalog_session (AsyncSession):
            Catalog session value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    result = await import_btb_documents(catalog_session)

    assert result.inserted is True
    assert result.item_count >= 100
    assert result.variant_count >= result.item_count
    assert result.policy_chunk_count >= 25
    assert result.incomplete_allergen_item_count > 0

    tenant_id = await catalog_session.scalar(
        select(Tenant.id).where(Tenant.name == BTB_TENANT_NAME)
    )
    assert tenant_id == result.tenant_id

    source_document_count = await catalog_session.scalar(
        select(func.count()).select_from(SourceDocument)
    )
    batch = await catalog_session.scalar(select(MenuImportBatch))
    menu_version = await catalog_session.get(MenuVersion, result.menu_version_id)
    policy_chunk_count = await catalog_session.scalar(select(func.count()).select_from(PolicyChunk))

    assert source_document_count == 3
    assert batch is not None
    assert batch.status == "published"
    assert menu_version is not None
    assert menu_version.status == "published"
    assert policy_chunk_count == result.policy_chunk_count


async def test_btb_import_is_idempotent(catalog_session: AsyncSession) -> None:
    """Verify that BTB import is idempotent.

    Args:
        catalog_session (AsyncSession):
            Catalog session value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    first = await import_btb_documents(catalog_session)
    second = await import_btb_documents(catalog_session)

    assert first.inserted is True
    assert second.inserted is False
    assert second.menu_version_id == first.menu_version_id

    menu_version_count = await catalog_session.scalar(select(func.count()).select_from(MenuVersion))
    source_document_count = await catalog_session.scalar(
        select(func.count()).select_from(SourceDocument)
    )

    assert menu_version_count == 1
    assert source_document_count == 3


async def test_published_catalog_loader_preserves_safety_boundary(
    catalog_session: AsyncSession,
) -> None:
    """Verify that published catalog loader preserves safety boundary.

    Args:
        catalog_session (AsyncSession):
            Catalog session value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    result = await import_btb_documents(catalog_session)
    views = await load_published_catalog_item_views_for_tenant(
        catalog_session,
        result.tenant_id,
    )

    assert len(views) >= result.item_count
    incomplete_views = [view for view in views if not view.allergen_data_complete]
    assert incomplete_views

    filtered = filter_safe_items(
        incomplete_views,
        CustomerRestrictions(avoid_allergens={AllergenCode.TREE_NUT}, modes=set()),
    )

    assert filtered.safe_items == []
    assert {decision.reason for decision in filtered.decisions} == {"EXCLUDED_INCOMPLETE_DATA"}


async def test_pesto_items_carry_tree_nut_risk(
    catalog_session: AsyncSession,
) -> None:
    """Verify that pesto items carry tree nut risk.

    Args:
        catalog_session (AsyncSession):
            Catalog session value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    await import_btb_documents(catalog_session)

    pesto_item = await catalog_session.scalar(
        select(CatalogItem)
        .where(CatalogItem.display_name == "Veg Pesto Pizza")
        .options()
    )
    assert pesto_item is not None

    assertion_count = await catalog_session.scalar(
        select(func.count())
        .select_from(CatalogItemAllergenAssertion)
        .join(CatalogItem)
        .where(CatalogItem.display_name == "Veg Pesto Pizza")
        .where(CatalogItemAllergenAssertion.assertion_type.in_(["CONTAINS", "MAY_CONTAIN"]))
    )
    variant_count = await catalog_session.scalar(
        select(func.count())
        .select_from(CatalogItemVariant)
        .join(CatalogItem)
        .where(CatalogItem.display_name == "Veg Pesto Pizza")
    )

    assert assertion_count is not None
    assert assertion_count > 0
    assert variant_count == 1


async def test_selected_modifier_allergen_risk_is_projected_into_safe_view(
    catalog_session: AsyncSession,
) -> None:
    """Verify that selected modifier allergen risk is projected into safe view.

    Args:
        catalog_session (AsyncSession):
            Catalog session value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    result = await import_btb_documents(catalog_session)
    latte_variant = await catalog_session.scalar(
        select(CatalogItemVariant)
        .join(CatalogItem)
        .where(CatalogItem.display_name == "Latte")
        .order_by(CatalogItemVariant.id)
    )
    almond_milk = await catalog_session.scalar(
        select(ModifierOption).where(ModifierOption.name == "Almond Milk")
    )

    assert latte_variant is not None
    assert almond_milk is not None

    views = await load_published_catalog_item_views_for_tenant(
        catalog_session,
        result.tenant_id,
        [latte_variant.id],
        {latte_variant.id: [almond_milk.id]},
    )

    assert len(views) == 1
    assert "Almond Milk" in views[0].name
    assert AllergenCode.TREE_NUT in views[0].allergen_codes

    filtered = filter_safe_items(
        views,
        CustomerRestrictions(avoid_allergens={AllergenCode.TREE_NUT}, modes=set()),
    )

    assert filtered.safe_items == []
    assert filtered.decisions[0].reason == "EXCLUDED_ALLERGEN_TREE_NUT"


async def test_incomplete_selected_modifier_data_is_excluded_for_allergen_avoidance(
    catalog_session: AsyncSession,
) -> None:
    """Verify that incomplete selected modifier data is excluded for allergen avoidance.

    Args:
        catalog_session (AsyncSession):
            Catalog session value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    result = await import_btb_documents(catalog_session)
    latte_variant = await catalog_session.scalar(
        select(CatalogItemVariant)
        .join(CatalogItem)
        .where(CatalogItem.display_name == "Latte")
        .order_by(CatalogItemVariant.id)
    )
    oat_milk = await catalog_session.scalar(
        select(ModifierOption).where(ModifierOption.name == "Oat Milk")
    )

    assert latte_variant is not None
    assert oat_milk is not None

    views = await load_published_catalog_item_views_for_tenant(
        catalog_session,
        result.tenant_id,
        [latte_variant.id],
        {latte_variant.id: [oat_milk.id]},
    )

    assert views[0].allergen_data_complete is False
    assert AllergenCode.GLUTEN in views[0].allergen_codes

    filtered = filter_safe_items(
        views,
        CustomerRestrictions(avoid_allergens={AllergenCode.GLUTEN}, modes=set()),
    )

    assert filtered.safe_items == []
    assert filtered.decisions[0].reason == EXCLUDED_INCOMPLETE_DATA


async def test_unknown_selected_modifier_is_treated_as_incomplete_data(
    catalog_session: AsyncSession,
) -> None:
    """Verify that unknown selected modifier is treated as incomplete data.

    Args:
        catalog_session (AsyncSession):
            Catalog session value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    result = await import_btb_documents(catalog_session)
    latte_variant = await catalog_session.scalar(
        select(CatalogItemVariant)
        .join(CatalogItem)
        .where(CatalogItem.display_name == "Latte")
        .order_by(CatalogItemVariant.id)
    )

    assert latte_variant is not None

    views = await load_published_catalog_item_views_for_tenant(
        catalog_session,
        result.tenant_id,
        [latte_variant.id],
        {latte_variant.id: [999_999]},
    )

    assert views[0].allergen_data_complete is False

    filtered = filter_safe_items(
        views,
        CustomerRestrictions(avoid_allergens={AllergenCode.PEANUT}, modes=set()),
    )

    assert filtered.safe_items == []
    assert filtered.decisions[0].reason == EXCLUDED_INCOMPLETE_DATA
