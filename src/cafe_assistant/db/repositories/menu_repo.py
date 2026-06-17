from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cafe_assistant.db.models import Ingredient, MenuItem
from cafe_assistant.domain.dietary import AllergenCode, DietaryMode, MenuItemView


async def load_menu_item_views_for_tenant(
    session: AsyncSession,
    tenant_id: int,
    item_ids: list[int] | None = None,
) -> list[MenuItemView]:
    statement = (
        select(MenuItem)
        .where(MenuItem.tenant_id == tenant_id)
        .options(
            selectinload(MenuItem.ingredients).selectinload(Ingredient.allergens),
            selectinload(MenuItem.dietary_tags),
        )
        .order_by(MenuItem.id)
    )
    if item_ids is not None:
        if not item_ids:
            return []
        statement = statement.where(MenuItem.id.in_(item_ids))

    result = await session.scalars(
        statement
    )

    return [_to_menu_item_view(item) for item in result.unique()]


def _to_menu_item_view(item: MenuItem) -> MenuItemView:
    allergen_codes = {
        _parse_allergen_code(allergen.code)
        for ingredient in item.ingredients
        for allergen in ingredient.allergens
    }
    dietary_tags = {_parse_dietary_mode(tag.code) for tag in item.dietary_tags}

    return MenuItemView(
        id=item.id,
        name=item.name,
        allergen_codes=allergen_codes,
        dietary_tags=dietary_tags,
        allergen_data_complete=item.allergen_data_complete,
        sugar_grams=float(item.sugar_grams) if item.sugar_grams is not None else None,
    )


def _parse_allergen_code(code: str) -> AllergenCode:
    try:
        return AllergenCode(code)
    except ValueError as exc:
        raise ValueError(f"Unknown allergen code in database: {code}") from exc


def _parse_dietary_mode(code: str) -> DietaryMode:
    try:
        return DietaryMode(code)
    except ValueError as exc:
        raise ValueError(f"Unknown dietary tag code in database: {code}") from exc
