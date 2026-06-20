"""Implementation module for dietary.
Contains typed helpers used by the cafe assistant backend runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class AllergenCode(StrEnum):
    """Enumeration of supported allergen code values."""
    PEANUT = "PEANUT"
    TREE_NUT = "TREE_NUT"
    DAIRY = "DAIRY"
    GLUTEN = "GLUTEN"
    SOY = "SOY"
    EGG = "EGG"
    FISH = "FISH"


class DietaryMode(StrEnum):
    """Enumeration of supported dietary mode values."""
    VEGAN = "VEGAN"
    VEGETARIAN = "VEGETARIAN"
    GLUTEN_FREE = "GLUTEN_FREE"


@dataclass(frozen=True, slots=True)
class CustomerRestrictions:
    """Container for customer restrictions behavior and data."""
    avoid_allergens: set[AllergenCode]
    modes: set[DietaryMode]
    prefer_low_sugar: bool = False


@dataclass(frozen=True, slots=True)
class MenuItemView:
    """Container for menu item view behavior and data."""
    id: int
    name: str
    allergen_codes: set[AllergenCode]
    dietary_tags: set[DietaryMode]
    allergen_data_complete: bool
    sugar_grams: float | None
    dietary_data_complete: bool = True


@dataclass(frozen=True, slots=True)
class FilterDecision:
    """Container for filter decision behavior and data."""
    item_id: int
    item_name: str
    included: bool
    reason: str


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Container for filter result behavior and data."""
    safe_items: list[MenuItemView]
    decisions: list[FilterDecision]


INCLUDED = "INCLUDED"
EXCLUDED_INCOMPLETE_DATA = "EXCLUDED_INCOMPLETE_DATA"
EXCLUDED_INCOMPLETE_DIETARY_DATA = "EXCLUDED_INCOMPLETE_DIETARY_DATA"

_ALLERGEN_REASON_BY_CODE = {
    allergen: f"EXCLUDED_ALLERGEN_{allergen.value}" for allergen in AllergenCode
}
_MODE_REASON_BY_CODE = {
    DietaryMode.VEGAN: "EXCLUDED_NOT_VEGAN",
    DietaryMode.VEGETARIAN: "EXCLUDED_NOT_VEGETARIAN",
    DietaryMode.GLUTEN_FREE: "EXCLUDED_NOT_GLUTEN_FREE",
}


def filter_safe_items(
    items: list[MenuItemView],
    restrictions: CustomerRestrictions,
) -> FilterResult:
    """Filter menu items according to hard allergen and dietary safety rules.

    Args:
        items (list[MenuItemView]):
            Menu item views evaluated or serialized by this function.
        restrictions (CustomerRestrictions):
            Customer allergen, dietary, and sugar preferences for the active turn.

    Returns:
        FilterResult:
            Safe items plus explicit include/exclude decisions for every input item.
    """
    decisions: list[FilterDecision] = []
    safe_items_with_index: list[tuple[int, MenuItemView]] = []

    for index, item in enumerate(items):
        reason = _exclusion_reason(item, restrictions)
        included = reason is None
        decisions.append(
            FilterDecision(
                item_id=item.id,
                item_name=item.name,
                included=included,
                reason=INCLUDED if included else reason,
            )
        )
        if included:
            safe_items_with_index.append((index, item))

    if restrictions.prefer_low_sugar:
        safe_items_with_index.sort(key=lambda indexed_item: _sugar_sort_key(*indexed_item))

    return FilterResult(
        safe_items=[item for _, item in safe_items_with_index],
        decisions=decisions,
    )


def _exclusion_reason(
    item: MenuItemView,
    restrictions: CustomerRestrictions,
) -> str | None:
    """Handle exclusion reason.

    Args:
        item (MenuItemView):
            Menu or catalog item being transformed, embedded, or evaluated.
        restrictions (CustomerRestrictions):
            Customer allergen, dietary, and sugar preferences for the active turn.

    Returns:
        str | None:
            Value produced for the caller according to the function contract.
    """
    if restrictions.avoid_allergens and not item.allergen_data_complete:
        return EXCLUDED_INCOMPLETE_DATA

    allergen_overlap = restrictions.avoid_allergens & item.allergen_codes
    if allergen_overlap:
        allergen = _first_allergen(allergen_overlap)
        return _ALLERGEN_REASON_BY_CODE[allergen]

    if restrictions.modes and not item.dietary_data_complete:
        return EXCLUDED_INCOMPLETE_DIETARY_DATA

    for mode in DietaryMode:
        if mode in restrictions.modes and mode not in item.dietary_tags:
            return _MODE_REASON_BY_CODE[mode]

    return None


def _first_allergen(allergens: set[AllergenCode]) -> AllergenCode:
    """Handle first allergen.

    Args:
        allergens (set[AllergenCode]):
            Allergen reference rows keyed by allergen code.

    Returns:
        AllergenCode:
            Value produced for the caller according to the function contract.
    """
    for allergen in AllergenCode:
        if allergen in allergens:
            return allergen
    raise ValueError("Allergen set unexpectedly contained no known allergen codes.")


def _sugar_sort_key(index: int, item: MenuItemView) -> tuple[int, float, int]:
    """Handle sugar sort key.

    Args:
        index (int):
            Index value required to perform this operation.
        item (MenuItemView):
            Menu or catalog item being transformed, embedded, or evaluated.

    Returns:
        tuple[int, float, int]:
            Value produced for the caller according to the function contract.
    """
    sugar = item.sugar_grams
    if sugar is None or not math.isfinite(sugar):
        return (1, 0.0, index)
    return (0, sugar, index)
