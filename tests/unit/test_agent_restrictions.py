"""Unit tests for deterministic agent restriction extraction.

These tests cover mixed positive and negated allergen statements because those
phrases feed the safety filter and durable/session memory merge. A negation for
one allergen must never erase a separate positive avoidance in the same turn.
"""

from __future__ import annotations

from cafe_assistant.agent.restrictions import extract_restrictions
from cafe_assistant.domain.dietary import AllergenCode, CustomerRestrictions


def _codes(message: str, stored: CustomerRestrictions | None = None) -> set[AllergenCode]:
    """Extract only allergen avoidance codes for concise assertions.

    Args:
        message (str):
            User message passed through the deterministic restriction extractor.
        stored (CustomerRestrictions | None):
            Optional prior session/profile restrictions used to test overrides.

    Returns:
        set[AllergenCode]:
            Active allergen avoidances after current-turn extraction and merge.
    """
    return extract_restrictions(message, stored).restrictions.avoid_allergens


def test_mixed_negation_keeps_later_positive_allergy() -> None:
    """Verify one negated allergen does not cancel a later positive allergen.

    Args:
        None:
            The test builds its own message and expected allergen set.

    Returns:
        None:
            Failed expectations raise pytest assertion errors.
    """
    assert _codes("I am not allergic to peanuts but I am allergic to dairy.") == {
        AllergenCode.DAIRY
    }


def test_no_specific_allergy_can_coexist_with_other_avoidance() -> None:
    """Verify a specific allergy negation can coexist with a new avoidance.

    Args:
        None:
            The test builds stored restrictions and a current-turn message.

    Returns:
        None:
            Failed expectations raise pytest assertion errors.
    """
    stored = CustomerRestrictions(avoid_allergens={AllergenCode.PEANUT}, modes=set())

    assert _codes("no peanut allergy, avoid gluten", stored) == {AllergenCode.GLUTEN}


def test_current_turn_removes_stored_allergen_and_adds_new_one() -> None:
    """Verify current-turn instructions override stored memory only where mentioned.

    Args:
        None:
            The test builds stored restrictions and a mixed override message.

    Returns:
        None:
            Failed expectations raise pytest assertion errors.
    """
    stored = CustomerRestrictions(
        avoid_allergens={AllergenCode.SOY, AllergenCode.PEANUT},
        modes=set(),
    )

    assert _codes("I don't avoid soy, but I can't have eggs", stored) == {
        AllergenCode.PEANUT,
        AllergenCode.EGG,
    }


def test_allergy_list_still_applies_to_multiple_allergens() -> None:
    """Verify comma-separated allergens stay inside one positive allergy context.

    Args:
        None:
            The test builds its own message and expected allergen set.

    Returns:
        None:
            Failed expectations raise pytest assertion errors.
    """
    assert _codes("I am allergic to peanuts, tree nuts, and dairy.") == {
        AllergenCode.PEANUT,
        AllergenCode.TREE_NUT,
        AllergenCode.DAIRY,
    }

def test_cannot_eat_phrase_marks_allergen_avoidance() -> None:
    """Verify food-intolerance wording still activates allergen avoidance.

    Args:
        None:
            The test builds its own message and expected allergen set.

    Returns:
        None:
            Failed expectations raise pytest assertion errors.
    """
    assert _codes("I cannot eat gluten. Can I have classic garlic bread?") == {
        AllergenCode.GLUTEN
    }
