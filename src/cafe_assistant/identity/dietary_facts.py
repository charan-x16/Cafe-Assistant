"""Serialization helpers for consent-gated dietary profile facts.

This module contains the shared conversion from deterministic restriction
extraction into JSON-safe durable profile fields. It intentionally has no
authentication, network, or model dependency so both account login flows and the
memory write gate can use the same health-data representation.
"""

from __future__ import annotations

from cafe_assistant.domain.dietary import CustomerRestrictions


def restrictions_to_dietary_facts(
    restrictions: CustomerRestrictions,
) -> dict[str, object]:
    """Convert explicit restrictions into durable dietary fact JSON.

    Args:
        restrictions (CustomerRestrictions):
            Current-turn restrictions that were explicitly mentioned by the
            customer and approved for durable persistence by the consent gate.

    Returns:
        dict[str, object]:
            JSON-serializable dietary facts suitable for profile storage. Empty
            restriction sets produce an empty dictionary.
    """
    facts: dict[str, object] = {}
    if restrictions.avoid_allergens:
        facts["avoid_allergens"] = sorted(
            allergen.value for allergen in restrictions.avoid_allergens
        )
    if restrictions.modes:
        facts["modes"] = sorted(mode.value for mode in restrictions.modes)
    if restrictions.prefer_low_sugar:
        facts["prefer_low_sugar"] = True
    return facts