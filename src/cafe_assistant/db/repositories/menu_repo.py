"""Implementation module for menu repo.
Contains typed helpers used by the cafe assistant backend runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cafe_assistant.db.models import (
    CatalogItem,
    CatalogItemAllergenAssertion,
    CatalogItemDietaryAssertion,
    CatalogItemVariant,
    Ingredient,
    Menu,
    MenuItem,
    MenuVersion,
    ModifierOption,
    ModifierOptionAllergenAssertion,
    ModifierOptionDietaryAssertion,
)
from cafe_assistant.domain.catalog_safety import (
    AllergenAssertionType,
    CatalogAllergenAssertion,
    CatalogDietaryAssertion,
    CatalogModifierSafety,
    DietaryAssertionType,
    merge_catalog_modifier_safety,
    suitable_dietary_tags,
    unsafe_allergen_codes,
)
from cafe_assistant.domain.dietary import AllergenCode, DietaryMode, MenuItemView


@dataclass(frozen=True, slots=True)
class MenuBrowseItem:
    """Safety-filterable menu item with its catalog or legacy section metadata.

    Args:
        item (MenuItemView):
            Menu item view consumed by the deterministic safety filter.
        category_name (str):
            Human-readable leaf category name from the source menu.
        category_path (str):
            Hierarchical category path such as `Beverages > Coffees`.

    Returns:
        None:
            Dataclass instances are value objects returned by repository helpers.
    """

    item: MenuItemView
    category_name: str
    category_path: str

async def load_menu_browse_items_for_tenant(
    session: AsyncSession,
    tenant_id: int,
) -> list[MenuBrowseItem]:
    """Load ordered menu browse entries with section metadata for one tenant.

    The helper prefers the imported production catalog and falls back to the
    legacy menu table used by tests. It does not decide safety itself; callers
    must still run the returned `MenuItemView` objects through the deterministic
    filter before showing names to a customer.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads.
        tenant_id (int):
            Tenant identifier used to scope catalog or legacy menu rows.

    Returns:
        list[MenuBrowseItem]:
            Ordered menu entries with their source category metadata.
    """
    catalog_entries = await _load_published_catalog_browse_items(session, tenant_id)
    if catalog_entries:
        return catalog_entries
    return await _load_legacy_menu_browse_items(session, tenant_id)


async def _load_published_catalog_browse_items(
    session: AsyncSession,
    tenant_id: int,
) -> list[MenuBrowseItem]:
    """Load published catalog variants with category metadata for browsing.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped catalog reads.
        tenant_id (int):
            Tenant identifier used to scope the published catalog.

    Returns:
        list[MenuBrowseItem]:
            Catalog variant views paired with category names and paths.
    """
    statement = (
        select(CatalogItemVariant)
        .join(CatalogItem)
        .join(MenuVersion)
        .join(Menu)
        .where(Menu.tenant_id == tenant_id)
        .where(MenuVersion.status == "published")
        .where(CatalogItem.is_available.is_(True))
        .where(CatalogItemVariant.is_available.is_(True))
        .options(
            selectinload(CatalogItemVariant.nutrition),
            selectinload(CatalogItemVariant.catalog_item).selectinload(CatalogItem.category),
            selectinload(CatalogItemVariant.catalog_item)
            .selectinload(CatalogItem.allergen_assertions)
            .selectinload(CatalogItemAllergenAssertion.allergen),
            selectinload(CatalogItemVariant.catalog_item)
            .selectinload(CatalogItem.dietary_assertions)
            .selectinload(CatalogItemDietaryAssertion.dietary_tag),
        )
        .order_by(CatalogItem.sort_order, CatalogItemVariant.sort_order, CatalogItemVariant.id)
    )
    result = await session.scalars(statement)
    entries: list[MenuBrowseItem] = []
    for variant in result.unique():
        category = variant.catalog_item.category
        category_name = category.name if category is not None else "Other"
        category_path = category.path if category is not None else category_name
        entries.append(
            MenuBrowseItem(
                item=_catalog_variant_to_menu_item_view(variant),
                category_name=category_name,
                category_path=category_path,
            )
        )
    return entries


async def _load_legacy_menu_browse_items(
    session: AsyncSession,
    tenant_id: int,
) -> list[MenuBrowseItem]:
    """Load legacy menu rows with category metadata for browsing.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped legacy menu reads.
        tenant_id (int):
            Tenant identifier used to scope legacy menu rows.

    Returns:
        list[MenuBrowseItem]:
            Legacy menu item views paired with their flat category names.
    """
    statement = (
        select(MenuItem)
        .where(MenuItem.tenant_id == tenant_id)
        .where(MenuItem.is_available.is_(True))
        .options(
            selectinload(MenuItem.ingredients).selectinload(Ingredient.allergens),
            selectinload(MenuItem.dietary_tags),
        )
        .order_by(MenuItem.category, MenuItem.id)
    )
    result = await session.scalars(statement)
    return [
        MenuBrowseItem(
            item=_to_menu_item_view(item),
            category_name=item.category,
            category_path=item.category,
        )
        for item in result.unique()
    ]

async def load_menu_item_views_for_tenant(
    session: AsyncSession,
    tenant_id: int,
    item_ids: list[int] | None = None,
) -> list[MenuItemView]:
    """Load legacy menu rows as safety-filterable views for one tenant.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        item_ids (list[int] | None):
            Item ids value required to perform this operation.

    Returns:
        list[MenuItemView]:
            Legacy menu item views scoped to the tenant.
    """
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
    """Convert menu item view.

    Args:
        item (MenuItem):
            Menu or catalog item being transformed, embedded, or evaluated.

    Returns:
        MenuItemView:
            Value produced for the caller according to the function contract.
    """
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


async def load_published_catalog_item_views_for_tenant(
    session: AsyncSession,
    tenant_id: int,
    variant_ids: list[int] | None = None,
    selected_modifier_option_ids_by_variant: dict[int, list[int]] | None = None,
) -> list[MenuItemView]:
    """Load published catalog variants as safety-filterable views for one tenant.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        tenant_id (int):
            Tenant identifier used to scope database and vector-store operations.
        variant_ids (list[int] | None):
            Variant ids value required to perform this operation.
        selected_modifier_option_ids_by_variant (dict[int, list[int]] | None):
            Selected modifier option IDs keyed by variant ID.

    Returns:
        list[MenuItemView]:
            Published catalog item views scoped to the tenant.
    """
    statement = (
        select(CatalogItemVariant)
        .join(CatalogItem)
        .join(MenuVersion)
        .join(Menu)
        .where(Menu.tenant_id == tenant_id)
        .where(MenuVersion.status == "published")
        .where(CatalogItem.is_available.is_(True))
        .where(CatalogItemVariant.is_available.is_(True))
        .options(
            selectinload(CatalogItemVariant.nutrition),
            selectinload(CatalogItemVariant.catalog_item)
            .selectinload(CatalogItem.allergen_assertions)
            .selectinload(CatalogItemAllergenAssertion.allergen),
            selectinload(CatalogItemVariant.catalog_item)
            .selectinload(CatalogItem.dietary_assertions)
            .selectinload(CatalogItemDietaryAssertion.dietary_tag),
        )
        .order_by(CatalogItem.sort_order, CatalogItemVariant.sort_order, CatalogItemVariant.id)
    )
    if variant_ids is not None:
        if not variant_ids:
            return []
        statement = statement.where(CatalogItemVariant.id.in_(variant_ids))

    result = await session.scalars(statement)
    variants = list(result.unique())
    selected_modifier_safety_by_variant = await _load_selected_modifier_safety(
        session,
        selected_modifier_option_ids_by_variant or {},
    )
    return [
        _catalog_variant_to_menu_item_view(
            variant,
            selected_modifier_safety_by_variant.get(variant.id, []),
        )
        for variant in variants
    ]


def _catalog_variant_to_menu_item_view(
    variant: CatalogItemVariant,
    selected_modifier_safety: list[CatalogModifierSafety] | None = None,
) -> MenuItemView:
    """Handle catalog variant to menu item view.

    Args:
        variant (CatalogItemVariant):
            Catalog variant row converted into a retrievable menu view.
        selected_modifier_safety (list[CatalogModifierSafety] | None):
            Resolved safety data for selected modifier options.

    Returns:
        MenuItemView:
            Value produced for the caller according to the function contract.
    """
    item = variant.catalog_item
    allergen_assertions = [
        CatalogAllergenAssertion(
            code=_parse_allergen_code(assertion.allergen.code),
            assertion_type=_parse_allergen_assertion_type(assertion.assertion_type),
        )
        for assertion in item.allergen_assertions
    ]
    dietary_assertions = [
        CatalogDietaryAssertion(
            code=_parse_dietary_mode(assertion.dietary_tag.code),
            assertion_type=_parse_dietary_assertion_type(assertion.assertion_type),
        )
        for assertion in item.dietary_assertions
    ]
    sugar_grams = None
    if variant.nutrition is not None and variant.nutrition.sugar_grams is not None:
        sugar_grams = float(variant.nutrition.sugar_grams)

    base_view = MenuItemView(
        id=variant.id,
        name=_catalog_variant_display_name(item.display_name, variant.name),
        allergen_codes=unsafe_allergen_codes(allergen_assertions),
        dietary_tags=suitable_dietary_tags(dietary_assertions),
        allergen_data_complete=item.allergen_data_complete,
        sugar_grams=sugar_grams,
        dietary_data_complete=item.dietary_data_complete,
    )
    return merge_catalog_modifier_safety(base_view, selected_modifier_safety or [])


async def _load_selected_modifier_safety(
    session: AsyncSession,
    selected_modifier_option_ids_by_variant: dict[int, list[int]],
) -> dict[int, list[CatalogModifierSafety]]:
    """Load selected modifier safety.

    Args:
        session (AsyncSession):
            Async SQLAlchemy session used for tenant-scoped database reads and writes.
        selected_modifier_option_ids_by_variant (dict[int, list[int]]):
            Selected modifier option IDs keyed by variant ID.

    Returns:
        dict[int, list[CatalogModifierSafety]]:
            Loaded records or projected domain values matching the requested scope.
    """
    option_ids = {
        option_id
        for option_ids_for_variant in selected_modifier_option_ids_by_variant.values()
        for option_id in option_ids_for_variant
    }
    if not option_ids:
        return {}

    result = await session.scalars(
        select(ModifierOption)
        .where(ModifierOption.id.in_(option_ids))
        .options(
            selectinload(ModifierOption.allergen_assertions).selectinload(
                ModifierOptionAllergenAssertion.allergen
            ),
            selectinload(ModifierOption.dietary_assertions).selectinload(
                ModifierOptionDietaryAssertion.dietary_tag
            ),
        )
    )
    options_by_id = {option.id: option for option in result.unique()}
    safety_by_variant: dict[int, list[CatalogModifierSafety]] = {}
    for variant_id, modifier_option_ids in selected_modifier_option_ids_by_variant.items():
        safety_by_variant[variant_id] = [
            _modifier_option_to_safety(options_by_id[option_id])
            if option_id in options_by_id
            else _unknown_modifier_safety(option_id)
            for option_id in modifier_option_ids
        ]
    return safety_by_variant


def _modifier_option_to_safety(option: ModifierOption) -> CatalogModifierSafety:
    """Handle modifier option to safety.

    Args:
        option (ModifierOption):
            Option value required to perform this operation.

    Returns:
        CatalogModifierSafety:
            Value produced for the caller according to the function contract.
    """
    allergen_assertions = [
        CatalogAllergenAssertion(
            code=_parse_allergen_code(assertion.allergen.code),
            assertion_type=_parse_allergen_assertion_type(assertion.assertion_type),
        )
        for assertion in option.allergen_assertions
    ]
    dietary_assertions = [
        CatalogDietaryAssertion(
            code=_parse_dietary_mode(assertion.dietary_tag.code),
            assertion_type=_parse_dietary_assertion_type(assertion.assertion_type),
        )
        for assertion in option.dietary_assertions
    ]
    return CatalogModifierSafety(
        name=option.name,
        allergen_codes=unsafe_allergen_codes(allergen_assertions),
        dietary_tags=suitable_dietary_tags(dietary_assertions),
        allergen_data_complete=option.allergen_data_complete,
        dietary_data_complete=option.dietary_data_complete,
    )


def _unknown_modifier_safety(option_id: int) -> CatalogModifierSafety:
    """Handle unknown modifier safety.

    Args:
        option_id (int):
            Option id value required to perform this operation.

    Returns:
        CatalogModifierSafety:
            Value produced for the caller according to the function contract.
    """
    return CatalogModifierSafety(
        name=f"Unknown modifier option {option_id}",
        allergen_codes=set(),
        dietary_tags=set(),
        allergen_data_complete=False,
        dietary_data_complete=False,
    )


def _catalog_variant_display_name(item_name: str, variant_name: str) -> str:
    """Handle catalog variant display name.

    Args:
        item_name (str):
            Item name value required to perform this operation.
        variant_name (str):
            Variant name value required to perform this operation.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    if variant_name == "Default":
        return item_name
    if f"({variant_name})" in item_name:
        return item_name
    return f"{item_name} ({variant_name})"


def _parse_allergen_code(code: str) -> AllergenCode:
    """Parse allergen code.

    Args:
        code (str):
            Code value required to perform this operation.

    Returns:
        AllergenCode:
            Parsed values extracted from the source text or structured payload.
    """
    try:
        return AllergenCode(code)
    except ValueError as exc:
        raise ValueError(f"Unknown allergen code in database: {code}") from exc


def _parse_dietary_mode(code: str) -> DietaryMode:
    """Parse dietary mode.

    Args:
        code (str):
            Code value required to perform this operation.

    Returns:
        DietaryMode:
            Parsed values extracted from the source text or structured payload.
    """
    try:
        return DietaryMode(code)
    except ValueError as exc:
        raise ValueError(f"Unknown dietary tag code in database: {code}") from exc


def _parse_allergen_assertion_type(assertion_type: str) -> AllergenAssertionType:
    """Parse allergen assertion type.

    Args:
        assertion_type (str):
            Assertion type value required to perform this operation.

    Returns:
        AllergenAssertionType:
            Parsed values extracted from the source text or structured payload.
    """
    try:
        return AllergenAssertionType(assertion_type)
    except ValueError:
        return AllergenAssertionType.UNKNOWN


def _parse_dietary_assertion_type(assertion_type: str) -> DietaryAssertionType:
    """Parse dietary assertion type.

    Args:
        assertion_type (str):
            Assertion type value required to perform this operation.

    Returns:
        DietaryAssertionType:
            Parsed values extracted from the source text or structured payload.
    """
    try:
        return DietaryAssertionType(assertion_type)
    except ValueError:
        return DietaryAssertionType.UNKNOWN
