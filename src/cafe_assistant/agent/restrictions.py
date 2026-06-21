"""Deterministic extraction of customer restrictions from the active turn.

This module is deliberately rule-based. The agent can use model routing for
conversation classification, but allergen and dietary restrictions must be
extracted deterministically so safety behavior is stable, testable, and biased
away from false negatives. The extractor returns both active merged restrictions
for safety filtering and explicit current-turn restrictions for consent-gated
memory writes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cafe_assistant.domain.dietary import AllergenCode, CustomerRestrictions, DietaryMode

_ALLERGEN_TERMS = {
    AllergenCode.PEANUT: r"peanuts?",
    AllergenCode.TREE_NUT: r"tree\s+nuts?|almonds?",
    AllergenCode.DAIRY: r"dairy|milk|cheese|butter",
    AllergenCode.GLUTEN: r"gluten|wheat",
    AllergenCode.SOY: r"soy",
    AllergenCode.EGG: r"eggs?",
}
_ALLERGEN_PATTERNS = {
    allergen: re.compile(rf"\b(?:{term_pattern})\b", re.IGNORECASE)
    for allergen, term_pattern in _ALLERGEN_TERMS.items()
}
_CLAUSE_SPLIT_PATTERN = re.compile(r"\b(?:but|however|though|although)\b|[.;]", re.IGNORECASE)
_NEGATED_ALLERGEN_PATTERNS = {
    allergen: re.compile(
        rf"\b(?:not\s+allergic\s+to|no\s+allerg(?:y|ies)\s+to|"
        rf"do\s+not\s+avoid|don't\s+avoid)\b(?:(?!\bbut\b).){{0,80}}\b(?:{term_pattern})\b"
        rf"|\bno\s+(?:{term_pattern})\s+allerg(?:y|ies)\b",
        re.IGNORECASE,
    )
    for allergen, term_pattern in _ALLERGEN_TERMS.items()
}
_ALLERGY_PATTERN = re.compile(
    r"\b(allergic|allergy|allergies|avoid|can't have|cannot have|without)\b",
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
    """Restrictions and escalation flags extracted from one user message.

    Attributes:
        restrictions (CustomerRestrictions):
            Active restrictions after merging stored memory with current-turn
            additions and negations. This value is used for safety filtering.
        current_turn_restrictions (CustomerRestrictions):
            Only positive health/dietary facts explicitly mentioned in the
            current message. This value is eligible for consent-gated durable
            persistence and intentionally excludes stored facts.
        medical_question (bool):
            True when the user asked for medical advice that the agent should refuse.
    """

    restrictions: CustomerRestrictions
    current_turn_restrictions: CustomerRestrictions
    medical_question: bool


def extract_restrictions(
    message: str,
    stored: CustomerRestrictions | None = None,
) -> RestrictionExtraction:
    """Extract active and current-turn restrictions from a user message.

    Args:
        message (str):
            Current user message. This is treated as untrusted text and parsed
            only with deterministic regex rules.
        stored (CustomerRestrictions | None):
            Session/profile restrictions remembered before this turn. Explicit
            current-turn negations remove matching stored allergen avoidances;
            explicit current-turn avoidances add matching allergens.

    Returns:
        RestrictionExtraction:
            Active merged restrictions, explicit current-turn facts for the
            write-gate, and a medical-escalation flag.
    """
    avoid_allergens = set(stored.avoid_allergens) if stored is not None else set()
    modes = set(stored.modes) if stored is not None else set()
    prefer_low_sugar = stored.prefer_low_sugar if stored is not None else False

    current_avoid_allergens: set[AllergenCode] = set()
    current_modes: set[DietaryMode] = set()
    current_prefer_low_sugar = False

    for clause in _allergen_clauses(message):
        clause_allergens = {
            allergen
            for allergen, pattern in _ALLERGEN_PATTERNS.items()
            if pattern.search(clause)
        }
        if not clause_allergens:
            continue

        negated_allergens = {
            allergen
            for allergen in clause_allergens
            if _NEGATED_ALLERGEN_PATTERNS[allergen].search(clause)
        }
        avoid_allergens.difference_update(negated_allergens)

        if _ALLERGY_PATTERN.search(clause):
            positive_allergens = clause_allergens - negated_allergens
            avoid_allergens.update(positive_allergens)
            current_avoid_allergens.update(positive_allergens)

    if _VEGAN_PATTERN.search(message):
        modes.add(DietaryMode.VEGAN)
        modes.add(DietaryMode.VEGETARIAN)
        current_modes.add(DietaryMode.VEGAN)
        current_modes.add(DietaryMode.VEGETARIAN)
    if _VEGETARIAN_PATTERN.search(message):
        modes.add(DietaryMode.VEGETARIAN)
        current_modes.add(DietaryMode.VEGETARIAN)
    if _GLUTEN_FREE_PATTERN.search(message):
        modes.add(DietaryMode.GLUTEN_FREE)
        current_modes.add(DietaryMode.GLUTEN_FREE)
    if _LOW_SUGAR_PATTERN.search(message):
        prefer_low_sugar = True
        current_prefer_low_sugar = True

    return RestrictionExtraction(
        restrictions=CustomerRestrictions(
            avoid_allergens=avoid_allergens,
            modes=modes,
            prefer_low_sugar=prefer_low_sugar,
        ),
        current_turn_restrictions=CustomerRestrictions(
            avoid_allergens=current_avoid_allergens,
            modes=current_modes,
            prefer_low_sugar=current_prefer_low_sugar,
        ),
        medical_question=bool(_MEDICAL_PATTERN.search(message)),
    )


def _allergen_clauses(message: str) -> list[str]:
    """Split a message on contrast boundaries relevant to allergen statements.

    Args:
        message (str):
            Current user message to split into local allergen contexts.

    Returns:
        list[str]:
            Non-empty clauses. Commas and `and` are intentionally preserved so
            phrases such as "allergic to peanuts, tree nuts, and dairy" still
            apply one allergy statement to the full allergen list.
    """
    return [clause.strip() for clause in _CLAUSE_SPLIT_PATTERN.split(message) if clause.strip()]