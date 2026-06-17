from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cafe_assistant.db.base import Base
from cafe_assistant.db.models import Tenant
from cafe_assistant.db.repositories.menu_repo import load_menu_item_views_for_tenant
from cafe_assistant.domain.dietary import (
    EXCLUDED_INCOMPLETE_DATA,
    INCLUDED,
    AllergenCode,
    CustomerRestrictions,
    DietaryMode,
    MenuItemView,
    filter_safe_items,
)
from scripts.seed_menu import TENANT_NAME, seed_database


def _menu_item_from_payload(payload: dict[str, Any]) -> MenuItemView:
    return MenuItemView(
        id=payload["id"],
        name=payload["name"],
        allergen_codes={AllergenCode(code) for code in payload["allergen_codes"]},
        dietary_tags={DietaryMode(code) for code in payload["dietary_tags"]},
        allergen_data_complete=payload["allergen_data_complete"],
        sugar_grams=payload["sugar_grams"],
    )


DATASET_PATH = Path(__file__).parents[2] / "evals" / "datasets" / "allergen_cases.json"
DATASET = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
ALL_ITEMS_BY_ID = {
    item.id: item for item in (_menu_item_from_payload(payload) for payload in DATASET["items"])
}


def _restrictions_from_payload(payload: dict[str, Any]) -> CustomerRestrictions:
    return CustomerRestrictions(
        avoid_allergens={AllergenCode(code) for code in payload["avoid_allergens"]},
        modes={DietaryMode(code) for code in payload["modes"]},
        prefer_low_sugar=payload["prefer_low_sugar"],
    )


def _items_for_case(case: dict[str, Any]) -> list[MenuItemView]:
    item_ids = case.get("item_ids", [item["id"] for item in DATASET["items"]])
    return [ALL_ITEMS_BY_ID[item_id] for item_id in item_ids]


@pytest.mark.parametrize("case", DATASET["cases"], ids=lambda case: case["id"])
def test_labeled_allergen_cases(case: dict[str, Any]) -> None:
    items = _items_for_case(case)
    restrictions = _restrictions_from_payload(case["restrictions"])
    expected_safe_ids = case["expected_safe_ids"]
    unsafe_item_ids = set(case["unsafe_item_ids"])
    input_item_ids = {item.id for item in items}

    assert set(expected_safe_ids).isdisjoint(unsafe_item_ids)
    assert set(expected_safe_ids) | unsafe_item_ids == input_item_ids

    result = filter_safe_items(items, restrictions)
    repeated_result = filter_safe_items(items, restrictions)

    safe_ids = [item.id for item in result.safe_items]
    assert safe_ids == expected_safe_ids
    assert repeated_result == result
    assert unsafe_item_ids.isdisjoint(safe_ids)

    decisions_by_id = {decision.item_id: decision for decision in result.decisions}
    assert set(decisions_by_id) == input_item_ids
    assert [decision.item_id for decision in result.decisions] == [item.id for item in items]

    for item_id in expected_safe_ids:
        assert decisions_by_id[item_id].included is True
        assert decisions_by_id[item_id].reason == INCLUDED

    for item_id in unsafe_item_ids:
        assert decisions_by_id[item_id].included is False
        assert decisions_by_id[item_id].reason != INCLUDED

    for item_id, reason in case.get("expected_reasons", {}).items():
        assert decisions_by_id[int(item_id)].reason == reason


def test_labeled_cases_have_zero_false_negatives() -> None:
    for case in DATASET["cases"]:
        result = filter_safe_items(
            _items_for_case(case),
            _restrictions_from_payload(case["restrictions"]),
        )
        safe_ids = {item.id for item in result.safe_items}

        assert safe_ids.isdisjoint(case["unsafe_item_ids"]), case["id"]


def test_incomplete_allergen_data_is_excluded_for_any_allergen_avoidance() -> None:
    incomplete_items = [
        MenuItemView(
            id=101,
            name="Unknown Pastry",
            allergen_codes=set(),
            dietary_tags={DietaryMode.VEGETARIAN},
            allergen_data_complete=False,
            sugar_grams=18.0,
        ),
        MenuItemView(
            id=102,
            name="Partially Labeled Sandwich",
            allergen_codes={AllergenCode.GLUTEN},
            dietary_tags=set(),
            allergen_data_complete=False,
            sugar_grams=None,
        ),
    ]
    allergens = list(AllergenCode)

    for item in incomplete_items:
        for subset_size in range(1, len(allergens) + 1):
            for allergen_subset in combinations(allergens, subset_size):
                result = filter_safe_items(
                    [item],
                    CustomerRestrictions(
                        avoid_allergens=set(allergen_subset),
                        modes=set(),
                        prefer_low_sugar=False,
                    ),
                )

                assert result.safe_items == []
                assert result.decisions[0].reason == EXCLUDED_INCOMPLETE_DATA


def test_sugar_preference_never_excludes_items() -> None:
    items = _items_for_case({"item_ids": [2, 5, 12, 14]})
    unrestricted = CustomerRestrictions(
        avoid_allergens=set(),
        modes=set(),
        prefer_low_sugar=False,
    )
    low_sugar_preferred = CustomerRestrictions(
        avoid_allergens=set(),
        modes=set(),
        prefer_low_sugar=True,
    )

    unsorted_result = filter_safe_items(items, unrestricted)
    sorted_result = filter_safe_items(items, low_sugar_preferred)

    assert {item.id for item in sorted_result.safe_items} == {
        item.id for item in unsorted_result.safe_items
    }
    assert [item.id for item in sorted_result.safe_items] == [2, 5, 12, 14]


async def test_load_menu_item_views_for_tenant_aggregates_seed_data() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    try:
        async with session_factory() as session:
            await seed_database(session)
            tenant_id = await session.scalar(select(Tenant.id).where(Tenant.name == TENANT_NAME))
            assert tenant_id is not None

            views = await load_menu_item_views_for_tenant(session, tenant_id)

        views_by_name = {view.name: view for view in views}
        assert len(views) == 16
        assert AllergenCode.DAIRY in views_by_name["Cappuccino"].allergen_codes
        assert DietaryMode.GLUTEN_FREE in views_by_name["Cappuccino"].dietary_tags
        assert views_by_name["Turkey Pesto Panini"].allergen_data_complete is False
        assert AllergenCode.GLUTEN in views_by_name["Turkey Pesto Panini"].allergen_codes
        assert views_by_name["Seasonal Berry Danish"].allergen_codes == set()
        assert views_by_name["Seasonal Berry Danish"].allergen_data_complete is False
    finally:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await engine.dispose()
