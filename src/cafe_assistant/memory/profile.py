from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.repositories.profile_repo import StoredProfile, load_stored_profile
from cafe_assistant.domain.dietary import AllergenCode, CustomerRestrictions, DietaryMode
from cafe_assistant.memory.session import SessionState


@dataclass(frozen=True, slots=True)
class DurableEvent:
    type: str
    payload: dict[str, object]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DurableProfile:
    customer_id: int
    tenant_id: int
    preferences: dict[str, object] = field(default_factory=dict)
    dietary_facts: dict[str, object] = field(default_factory=dict)
    consent_at: datetime | None = None
    recent_events: list[DurableEvent] = field(default_factory=list)

    @property
    def restrictions(self) -> CustomerRestrictions:
        return dietary_facts_to_restrictions(self.dietary_facts)


@dataclass(frozen=True, slots=True)
class AgentMemoryContext:
    session_state: SessionState
    durable_profile: DurableProfile | None
    preferences: dict[str, object]


async def load_durable_profile(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int | None,
) -> DurableProfile | None:
    if customer_id is None:
        return None
    stored = await load_stored_profile(
        session,
        tenant_id=tenant_id,
        customer_id=customer_id,
    )
    if stored is None:
        return None
    return _to_durable_profile(stored)


def merge_profile_with_session(
    *,
    session_state: SessionState,
    durable_profile: DurableProfile | None,
) -> AgentMemoryContext:
    if durable_profile is None:
        return AgentMemoryContext(
            session_state=session_state,
            durable_profile=None,
            preferences={},
        )

    merged_restrictions = merge_restrictions(
        durable_profile.restrictions,
        session_state.restrictions,
    )
    return AgentMemoryContext(
        session_state=SessionState(
            restrictions=merged_restrictions,
            recent_turns=list(session_state.recent_turns),
        ),
        durable_profile=durable_profile,
        preferences=dict(durable_profile.preferences),
    )


def merge_restrictions(
    durable: CustomerRestrictions,
    session: CustomerRestrictions,
) -> CustomerRestrictions:
    return CustomerRestrictions(
        avoid_allergens=set(durable.avoid_allergens) | set(session.avoid_allergens),
        modes=set(durable.modes) | set(session.modes),
        prefer_low_sugar=durable.prefer_low_sugar or session.prefer_low_sugar,
    )


def dietary_facts_to_restrictions(facts: dict[str, object]) -> CustomerRestrictions:
    return CustomerRestrictions(
        avoid_allergens=_parse_allergens(facts.get("avoid_allergens", [])),
        modes=_parse_modes(facts.get("modes", [])),
        prefer_low_sugar=bool(facts.get("prefer_low_sugar", False)),
    )


def _parse_allergens(value: object) -> set[AllergenCode]:
    if not isinstance(value, list):
        return set()
    parsed: set[AllergenCode] = set()
    for code in value:
        try:
            parsed.add(AllergenCode(str(code)))
        except ValueError:
            continue
    return parsed


def _parse_modes(value: object) -> set[DietaryMode]:
    if not isinstance(value, list):
        return set()
    parsed: set[DietaryMode] = set()
    for code in value:
        try:
            parsed.add(DietaryMode(str(code)))
        except ValueError:
            continue
    return parsed


def _to_durable_profile(stored: StoredProfile) -> DurableProfile:
    return DurableProfile(
        customer_id=stored.customer_id,
        tenant_id=stored.tenant_id,
        preferences=stored.preferences,
        dietary_facts=stored.dietary_facts,
        consent_at=stored.consent_at,
        recent_events=[
            DurableEvent(
                type=event.type,
                payload=dict(event.payload or {}),
                created_at=event.created_at,
            )
            for event in stored.recent_events
        ],
    )
