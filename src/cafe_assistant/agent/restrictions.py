from __future__ import annotations

import re
from dataclasses import dataclass

from cafe_assistant.domain.dietary import AllergenCode, CustomerRestrictions, DietaryMode

_ALLERGEN_PATTERNS = {
    AllergenCode.PEANUT: re.compile(r"\b(peanut|peanuts)\b", re.IGNORECASE),
    AllergenCode.TREE_NUT: re.compile(r"\b(tree nut|tree nuts|almond|almonds)\b", re.IGNORECASE),
    AllergenCode.DAIRY: re.compile(r"\b(dairy|milk|cheese|butter)\b", re.IGNORECASE),
    AllergenCode.GLUTEN: re.compile(r"\b(gluten|wheat)\b", re.IGNORECASE),
    AllergenCode.SOY: re.compile(r"\b(soy)\b", re.IGNORECASE),
    AllergenCode.EGG: re.compile(r"\b(egg|eggs)\b", re.IGNORECASE),
}
_NEGATION_PATTERN = re.compile(
    r"\b(not allergic|no allergy|no allergies|do not avoid|don't avoid)\b",
    re.IGNORECASE,
)
_ALLERGY_PATTERN = re.compile(
    r"\b(allergic|allergy|avoid|can't have|cannot have|without)\b",
    re.IGNORECASE,
)
_LOW_SUGAR_PATTERN = re.compile(
    r"\b(diabetic|diabetes|low sugar|less sugar|lower sugar)\b",
    re.IGNORECASE,
)
_VEGAN_PATTERN = re.compile(r"\bvegan\b", re.IGNORECASE)
_VEGETARIAN_PATTERN = re.compile(r"\bvegetarian\b", re.IGNORECASE)
_GLUTEN_FREE_PATTERN = re.compile(r"\bgluten[- ]?free\b", re.IGNORECASE)
_MEDICAL_PATTERN = re.compile(
    r"\b(insulin|dose|dosage|medication|medicine|blood sugar|glucose|a1c|carb count|"
    r"carb counting|how many carbs should|medical advice)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RestrictionExtraction:
    restrictions: CustomerRestrictions
    medical_question: bool


def extract_restrictions(
    message: str,
    stored: CustomerRestrictions | None = None,
) -> RestrictionExtraction:
    avoid_allergens = set(stored.avoid_allergens) if stored is not None else set()
    modes = set(stored.modes) if stored is not None else set()
    prefer_low_sugar = stored.prefer_low_sugar if stored is not None else False

    lower_message = message.lower()
    has_negation = bool(_NEGATION_PATTERN.search(message))
    has_allergy_language = bool(_ALLERGY_PATTERN.search(message))

    for allergen, pattern in _ALLERGEN_PATTERNS.items():
        if not pattern.search(message):
            continue
        if has_negation:
            avoid_allergens.discard(allergen)
        elif has_allergy_language:
            avoid_allergens.add(allergen)

    if _VEGAN_PATTERN.search(message):
        modes.add(DietaryMode.VEGAN)
        modes.add(DietaryMode.VEGETARIAN)
    if _VEGETARIAN_PATTERN.search(message):
        modes.add(DietaryMode.VEGETARIAN)
    if _GLUTEN_FREE_PATTERN.search(message):
        modes.add(DietaryMode.GLUTEN_FREE)
    if _LOW_SUGAR_PATTERN.search(message):
        prefer_low_sugar = True

    return RestrictionExtraction(
        restrictions=CustomerRestrictions(
            avoid_allergens=avoid_allergens,
            modes=modes,
            prefer_low_sugar=prefer_low_sugar,
        ),
        medical_question=bool(_MEDICAL_PATTERN.search(lower_message)),
    )
