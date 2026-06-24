"""Consent-aware write gate for durable customer memory.

The write gate is the only path from agent-observed facts to durable profile
writes. UI preferences, such as milk preference, may be persisted for recognized
customers and stored in anonymous session memory. Health and dietary facts,
including allergies and diabetes-related low-sugar preference, require active
consent before durable persistence. The gate accepts only facts explicitly
mentioned in the current turn so merged durable/session facts are not silently
re-written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.repositories.consent_repo import DIETARY_HEALTH_SCOPE, has_active_consent
from cafe_assistant.db.repositories.profile_repo import (
    append_event,
    update_dietary_facts,
    update_preferences,
)
from cafe_assistant.domain.dietary import CustomerRestrictions
from cafe_assistant.identity.dietary_facts import restrictions_to_dietary_facts

_MILK_PREFERENCE_PATTERN = re.compile(
    r"\b(?:i\s+)?(?:like|love|prefer)\s+(oat|almond|soy|whole)\s+milk\b",
    re.IGNORECASE,
)


class CandidateWriteKind(StrEnum):
    """Kinds of profile writes that can be evaluated by the write gate."""

    PREFERENCE = "preference"
    DIETARY_FACT = "dietary_fact"


@dataclass(frozen=True, slots=True)
class CandidateWrite:
    """One proposed durable-memory write from the current turn.

    Attributes:
        kind (CandidateWriteKind):
            Category of write being evaluated.
        payload (dict[str, object]):
            Redacted-safe structured facts to merge into the profile when allowed.
        required_scope (str | None):
            Consent scope required before persistence, or None for auto-write preferences.
    """

    kind: CandidateWriteKind
    payload: dict[str, object]
    required_scope: str | None = None


@dataclass(frozen=True, slots=True)
class WriteGateResult:
    """Result of evaluating candidate writes against consent and tenant scope.

    Attributes:
        persisted (list[CandidateWrite]):
            Candidate writes that were saved to durable profile storage.
        skipped (list[CandidateWrite]):
            Candidate writes that were blocked by consent or tenant/customer lookup.
    """

    persisted: list[CandidateWrite]
    skipped: list[CandidateWrite]


def extract_preferences(message: str) -> dict[str, object]:
    """Extract non-health preferences that may be remembered automatically.

    Args:
        message (str):
            Current user message to inspect for UI-style preferences.

    Returns:
        dict[str, object]:
            Preference updates, currently limited to a normalized milk preference.
    """
    match = _MILK_PREFERENCE_PATTERN.search(message)
    if match is None:
        return {}
    return {"milk_preference": f"{match.group(1).lower()} milk"}


def classify_candidate_writes(
    *,
    message: str,
    current_turn_restrictions: CustomerRestrictions,
    current_turn_preferences: dict[str, object] | None = None,
) -> list[CandidateWrite]:
    """Classify current-turn facts into durable-memory write candidates.

    Args:
        message (str):
            Current user message. Used only to extract auto-write preferences
            when `current_turn_preferences` was not already supplied.
        current_turn_restrictions (CustomerRestrictions):
            Positive health/dietary facts explicitly mentioned in this turn.
            Stored or merged restrictions must not be passed here.
        current_turn_preferences (dict[str, object] | None):
            Optional preferences already extracted for the active turn.

    Returns:
        list[CandidateWrite]:
            Preference writes plus consent-required dietary writes proposed for persistence.
    """
    writes: list[CandidateWrite] = []
    preference_payload = (
        dict(current_turn_preferences)
        if current_turn_preferences is not None
        else extract_preferences(message)
    )
    if preference_payload:
        writes.append(
            CandidateWrite(
                kind=CandidateWriteKind.PREFERENCE,
                payload=preference_payload,
            )
        )

    dietary_facts = restrictions_to_dietary_facts(current_turn_restrictions)
    if dietary_facts:
        writes.append(
            CandidateWrite(
                kind=CandidateWriteKind.DIETARY_FACT,
                payload=dietary_facts,
                required_scope=DIETARY_HEALTH_SCOPE,
            )
        )
    return writes


async def persist_allowed_writes(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
    writes: list[CandidateWrite],
) -> WriteGateResult:
    """Persist candidate writes that pass tenant lookup and consent checks.

    Args:
        session (AsyncSession):
            Async database session used by profile and consent repositories.
        tenant_id (int):
            Tenant scope for the customer profile being updated.
        customer_id (int):
            Durable customer ID within the tenant.
        writes (list[CandidateWrite]):
            Current-turn candidate writes to evaluate.

    Returns:
        WriteGateResult:
            Lists of persisted and skipped writes for auditing by the caller.
    """
    persisted: list[CandidateWrite] = []
    skipped: list[CandidateWrite] = []

    for write in writes:
        if write.kind == CandidateWriteKind.PREFERENCE:
            saved = await update_preferences(
                session,
                tenant_id=tenant_id,
                customer_id=customer_id,
                updates=write.payload,
            )
            if saved:
                await append_event(
                    session,
                    tenant_id=tenant_id,
                    customer_id=customer_id,
                    event_type="preference_saved",
                    payload={"keys": sorted(write.payload)},
                )
                persisted.append(write)
            else:
                skipped.append(write)
            continue

        if write.required_scope is not None and not await has_active_consent(
            session,
            tenant_id=tenant_id,
            customer_id=customer_id,
            scope=write.required_scope,
        ):
            skipped.append(write)
            continue

        saved = await update_dietary_facts(
            session,
            tenant_id=tenant_id,
            customer_id=customer_id,
            updates=write.payload,
        )
        if saved:
            await append_event(
                session,
                tenant_id=tenant_id,
                customer_id=customer_id,
                event_type="dietary_facts_saved",
                payload={"keys": sorted(write.payload)},
            )
            persisted.append(write)
        else:
            skipped.append(write)

    return WriteGateResult(persisted=persisted, skipped=skipped)