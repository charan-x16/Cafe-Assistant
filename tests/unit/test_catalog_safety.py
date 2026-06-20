"""Tests for catalog safety.
Exercises expected behavior with deterministic fixtures and mocked providers where needed.
"""

from __future__ import annotations

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


def test_allergen_assertion_projection_is_conservative() -> None:
    """Verify that allergen assertion projection is conservative.

    Args:
        None.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    assertions = [
        CatalogAllergenAssertion(AllergenCode.PEANUT, AllergenAssertionType.CONTAINS),
        CatalogAllergenAssertion(AllergenCode.TREE_NUT, AllergenAssertionType.MAY_CONTAIN),
        CatalogAllergenAssertion(AllergenCode.DAIRY, AllergenAssertionType.CROSS_CONTACT_RISK),
        CatalogAllergenAssertion(AllergenCode.SOY, AllergenAssertionType.UNKNOWN),
        CatalogAllergenAssertion(AllergenCode.GLUTEN, AllergenAssertionType.DOES_NOT_CONTAIN),
    ]

    assert unsafe_allergen_codes(assertions) == {
        AllergenCode.PEANUT,
        AllergenCode.TREE_NUT,
        AllergenCode.DAIRY,
        AllergenCode.SOY,
    }


def test_dietary_assertion_projection_only_accepts_explicit_suitable_tags() -> None:
    """Verify that dietary assertion projection only accepts explicit suitable tags.

    Args:
        None.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    assertions = [
        CatalogDietaryAssertion(DietaryMode.VEGAN, DietaryAssertionType.SUITABLE),
        CatalogDietaryAssertion(DietaryMode.VEGETARIAN, DietaryAssertionType.ADAPTABLE),
        CatalogDietaryAssertion(DietaryMode.GLUTEN_FREE, DietaryAssertionType.UNKNOWN),
    ]

    assert suitable_dietary_tags(assertions) == {DietaryMode.VEGAN}


def test_modifier_safety_adds_risk_and_tightens_completeness() -> None:
    """Verify that modifier safety adds risk and tightens completeness.

    Args:
        None.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    item = MenuItemView(
        id=1,
        name="Latte",
        allergen_codes={AllergenCode.DAIRY},
        dietary_tags={DietaryMode.VEGETARIAN},
        allergen_data_complete=True,
        sugar_grams=8.0,
        dietary_data_complete=True,
    )
    selected_modifiers = [
        CatalogModifierSafety(
            name="Almond Milk",
            allergen_codes={AllergenCode.TREE_NUT},
            dietary_tags={DietaryMode.VEGAN, DietaryMode.VEGETARIAN},
            allergen_data_complete=True,
            dietary_data_complete=True,
        ),
        CatalogModifierSafety(
            name="Unverified Topping",
            allergen_codes=set(),
            dietary_tags=set(),
            allergen_data_complete=False,
            dietary_data_complete=False,
        ),
    ]

    merged = merge_catalog_modifier_safety(item, selected_modifiers)

    assert merged.name == "Latte + Almond Milk, Unverified Topping"
    assert merged.allergen_codes == {AllergenCode.DAIRY, AllergenCode.TREE_NUT}
    assert merged.dietary_tags == set()
    assert merged.allergen_data_complete is False
    assert merged.dietary_data_complete is False
