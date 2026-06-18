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
from cafe_assistant.identity.otp import restrictions_to_dietary_facts

_MILK_PREFERENCE_PATTERN = re.compile(
    r"\b(?:i\s+)?(?:like|love|prefer)\s+(oat|almond|soy|whole)\s+milk\b",
    re.IGNORECASE,
)


class CandidateWriteKind(StrEnum):
    PREFERENCE = "preference"
    DIETARY_FACT = "dietary_fact"


@dataclass(frozen=True, slots=True)
class CandidateWrite:
    kind: CandidateWriteKind
    payload: dict[str, object]
    required_scope: str | None = None


@dataclass(frozen=True, slots=True)
class WriteGateResult:
    persisted: list[CandidateWrite]
    skipped: list[CandidateWrite]


def classify_candidate_writes(
    *,
    message: str,
    restrictions: CustomerRestrictions,
) -> list[CandidateWrite]:
    writes: list[CandidateWrite] = []
    preference_payload = _extract_preferences(message)
    if preference_payload:
        writes.append(
            CandidateWrite(
                kind=CandidateWriteKind.PREFERENCE,
                payload=preference_payload,
            )
        )

    dietary_facts = restrictions_to_dietary_facts(restrictions)
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


def _extract_preferences(message: str) -> dict[str, object]:
    match = _MILK_PREFERENCE_PATTERN.search(message)
    if match is None:
        return {}
    return {"milk_preference": f"{match.group(1).lower()} milk"}
