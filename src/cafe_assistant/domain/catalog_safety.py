"""Implementation module for catalog safety.
Contains typed helpers used by the cafe assistant backend runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from cafe_assistant.domain.dietary import AllergenCode, DietaryMode, MenuItemView


class AllergenAssertionType(StrEnum):
    """Enumeration of supported allergen assertion type values."""
    CONTAINS = "CONTAINS"
    MAY_CONTAIN = "MAY_CONTAIN"
    CROSS_CONTACT_RISK = "CROSS_CONTACT_RISK"
    DOES_NOT_CONTAIN = "DOES_NOT_CONTAIN"
    UNKNOWN = "UNKNOWN"


class DietaryAssertionType(StrEnum):
    """Enumeration of supported dietary assertion type values."""
    SUITABLE = "SUITABLE"
    ADAPTABLE = "ADAPTABLE"
    NOT_SUITABLE = "NOT_SUITABLE"
    UNKNOWN = "UNKNOWN"


UNSAFE_ALLERGEN_ASSERTIONS = {
    AllergenAssertionType.CONTAINS,
    AllergenAssertionType.MAY_CONTAIN,
    AllergenAssertionType.CROSS_CONTACT_RISK,
    AllergenAssertionType.UNKNOWN,
}
SAFE_DIETARY_ASSERTIONS = {DietaryAssertionType.SUITABLE}


@dataclass(frozen=True, slots=True)
class CatalogAllergenAssertion:
    """Container for catalog allergen assertion behavior and data."""
    code: AllergenCode
    assertion_type: AllergenAssertionType


@dataclass(frozen=True, slots=True)
class CatalogDietaryAssertion:
    """Container for catalog dietary assertion behavior and data."""
    code: DietaryMode
    assertion_type: DietaryAssertionType


@dataclass(frozen=True, slots=True)
class CatalogModifierSafety:
    """Container for catalog modifier safety behavior and data."""
    name: str
    allergen_codes: set[AllergenCode]
    dietary_tags: set[DietaryMode]
    allergen_data_complete: bool
    dietary_data_complete: bool


def unsafe_allergen_codes(
    assertions: list[CatalogAllergenAssertion],
) -> set[AllergenCode]:
    """Project catalog allergen assertions into codes that must be treated unsafe.

    Args:
        assertions (list[CatalogAllergenAssertion]):
            Catalog safety assertions projected into domain codes.

    Returns:
        set[AllergenCode]:
            Allergen codes that must exclude an item for matching avoidances.
    """
    return {
        assertion.code
        for assertion in assertions
        if assertion.assertion_type in UNSAFE_ALLERGEN_ASSERTIONS
    }


def suitable_dietary_tags(
    assertions: list[CatalogDietaryAssertion],
) -> set[DietaryMode]:
    """Project catalog dietary assertions into explicitly suitable dietary modes.

    Args:
        assertions (list[CatalogDietaryAssertion]):
            Catalog safety assertions projected into domain codes.

    Returns:
        set[DietaryMode]:
            Dietary modes explicitly supported by the assertions.
    """
    return {
        assertion.code
        for assertion in assertions
        if assertion.assertion_type in SAFE_DIETARY_ASSERTIONS
    }


def merge_catalog_modifier_safety(
    base_item: MenuItemView,
    selected_modifiers: list[CatalogModifierSafety],
) -> MenuItemView:
    """Merge selected modifier safety data into a base menu item view.

    Args:
        base_item (MenuItemView):
            Menu item view before selected modifier safety data is merged.
        selected_modifiers (list[CatalogModifierSafety]):
            Selected modifier safety records applied to a base item.

    Returns:
        MenuItemView:
            Menu item view with modifier risks and completeness flags applied.
    """
    if not selected_modifiers:
        return base_item

    allergen_codes = set(base_item.allergen_codes)
    dietary_tags = set(base_item.dietary_tags)
    allergen_data_complete = base_item.allergen_data_complete
    dietary_data_complete = base_item.dietary_data_complete

    for modifier in selected_modifiers:
        allergen_codes |= modifier.allergen_codes
        allergen_data_complete = allergen_data_complete and modifier.allergen_data_complete
        dietary_data_complete = dietary_data_complete and modifier.dietary_data_complete
        if modifier.dietary_data_complete:
            dietary_tags &= modifier.dietary_tags
        else:
            dietary_tags = set()

    return MenuItemView(
        id=base_item.id,
        name=_name_with_modifiers(base_item.name, selected_modifiers),
        allergen_codes=allergen_codes,
        dietary_tags=dietary_tags,
        allergen_data_complete=allergen_data_complete,
        sugar_grams=base_item.sugar_grams,
        dietary_data_complete=dietary_data_complete,
    )


def _name_with_modifiers(
    base_name: str,
    selected_modifiers: list[CatalogModifierSafety],
) -> str:
    """Handle name with modifiers.

    Args:
        base_name (str):
            Menu item name before selected modifier labels are appended.
        selected_modifiers (list[CatalogModifierSafety]):
            Selected modifier safety records applied to a base item.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    if not selected_modifiers:
        return base_name
    modifier_names = ", ".join(modifier.name for modifier in selected_modifiers)
    return f"{base_name} + {modifier_names}"
