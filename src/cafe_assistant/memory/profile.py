"""Durable profile loading and merge helpers for agent memory.

Durable profile memory comes from tenant-scoped database repositories and may
include preferences, consent-gated dietary facts, and recent episodic events.
This module converts stored rows into immutable runtime objects and merges them
with tenant-scoped session memory. Episodic events are currently exposed for
profile inspection and future personalization, but they are not injected into
model prompts or safety decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.repositories.profile_repo import StoredProfile, load_stored_profile
from cafe_assistant.domain.dietary import AllergenCode, CustomerRestrictions, DietaryMode
from cafe_assistant.memory.session import SessionState


@dataclass(frozen=True, slots=True)
class DurableEvent:
    """Inspectable durable event associated with a customer profile.

    Attributes:
        type (str):
            Event category, such as `preference_saved` or `otp_upgrade`.
        payload (dict[str, object]):
            Structured event metadata. Sensitive values should already be minimized.
        created_at (datetime):
            Timestamp when the event was recorded.
    """

    type: str
    payload: dict[str, object]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DurableProfile:
    """Runtime view of a tenant-scoped durable customer profile.

    Attributes:
        customer_id (int):
            Durable customer ID within the tenant.
        tenant_id (int):
            Tenant that owns the profile.
        preferences (dict[str, object]):
            Auto-writable UI preferences stored for the customer.
        dietary_facts (dict[str, object]):
            Consent-gated health/dietary facts stored for the customer.
        consent_at (datetime | None):
            Time when dietary/health consent was granted, when present.
        recent_events (list[DurableEvent]):
            Recent durable events for inspection and debugging.
    """

    customer_id: int
    tenant_id: int
    preferences: dict[str, object] = field(default_factory=dict)
    dietary_facts: dict[str, object] = field(default_factory=dict)
    consent_at: datetime | None = None
    recent_events: list[DurableEvent] = field(default_factory=list)

    @property
    def restrictions(self) -> CustomerRestrictions:
        """Convert stored dietary facts into safety-filter restrictions.

        Args:
            None:
                This property reads the profile's stored dietary facts.

        Returns:
            CustomerRestrictions:
                Restrictions represented by durable consent-gated facts.
        """
        return dietary_facts_to_restrictions(self.dietary_facts)


@dataclass(frozen=True, slots=True)
class AgentMemoryContext:
    """Merged memory context consumed by the chat state machine.

    Attributes:
        session_state (SessionState):
            Session state after durable restrictions have been merged in.
        durable_profile (DurableProfile | None):
            Loaded durable profile for recognized customers, or None for anonymous sessions.
        preferences (dict[str, object]):
            Durable and session preferences merged for retrieval and composition.
    """

    session_state: SessionState
    durable_profile: DurableProfile | None
    preferences: dict[str, object]


async def load_durable_profile(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int | None,
) -> DurableProfile | None:
    """Load a customer's durable profile inside one tenant.

    Args:
        session (AsyncSession):
            Async database session used by the profile repository.
        tenant_id (int):
            Tenant that must own the customer profile.
        customer_id (int | None):
            Candidate durable customer ID. None means anonymous session.

    Returns:
        DurableProfile | None:
            Runtime durable profile when the tenant/customer pair exists, otherwise None.
    """
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
    """Merge durable profile memory with tenant-scoped session memory.

    Args:
        session_state (SessionState):
            Tenant-scoped session overlay loaded for the request.
        durable_profile (DurableProfile | None):
            Durable profile for a recognized customer, or None for anonymous chat.

    Returns:
        AgentMemoryContext:
            Restrictions and preferences ready for active-turn extraction and retrieval.
    """
    if durable_profile is None:
        return AgentMemoryContext(
            session_state=session_state,
            durable_profile=None,
            preferences=dict(session_state.preferences),
        )

    merged_restrictions = merge_restrictions(
        durable_profile.restrictions,
        session_state.restrictions,
    )
    merged_preferences = {
        **dict(durable_profile.preferences),
        **dict(session_state.preferences),
    }
    return AgentMemoryContext(
        session_state=SessionState(
            restrictions=merged_restrictions,
            preferences=dict(session_state.preferences),
            recent_turns=list(session_state.recent_turns),
        ),
        durable_profile=durable_profile,
        preferences=merged_preferences,
    )


def merge_restrictions(
    durable: CustomerRestrictions,
    session: CustomerRestrictions,
) -> CustomerRestrictions:
    """Combine durable and session restrictions for active-turn safety filtering.

    Args:
        durable (CustomerRestrictions):
            Consent-gated restrictions loaded from the durable profile.
        session (CustomerRestrictions):
            Short-lived restrictions learned during the current session.

    Returns:
        CustomerRestrictions:
            Union of allergen and dietary modes plus combined low-sugar preference.
    """
    return CustomerRestrictions(
        avoid_allergens=set(durable.avoid_allergens) | set(session.avoid_allergens),
        modes=set(durable.modes) | set(session.modes),
        prefer_low_sugar=durable.prefer_low_sugar or session.prefer_low_sugar,
    )


def dietary_facts_to_restrictions(facts: dict[str, object]) -> CustomerRestrictions:
    """Convert stored dietary facts into domain restrictions.

    Args:
        facts (dict[str, object]):
            Stored dietary profile JSON.

    Returns:
        CustomerRestrictions:
            Parsed restrictions with malformed or unknown values ignored.
    """
    return CustomerRestrictions(
        avoid_allergens=_parse_allergens(facts.get("avoid_allergens", [])),
        modes=_parse_modes(facts.get("modes", [])),
        prefer_low_sugar=bool(facts.get("prefer_low_sugar", False)),
    )


def _parse_allergens(value: object) -> set[AllergenCode]:
    """Parse allergen codes from profile JSON.

    Args:
        value (object):
            Stored JSON value expected to be a list of allergen code strings.

    Returns:
        set[AllergenCode]:
            Valid allergen enum values. Unknown values are dropped.
    """
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
    """Parse dietary modes from profile JSON.

    Args:
        value (object):
            Stored JSON value expected to be a list of dietary mode strings.

    Returns:
        set[DietaryMode]:
            Valid dietary mode enum values. Unknown values are dropped.
    """
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
    """Convert repository output into an immutable runtime profile.

    Args:
        stored (StoredProfile):
            Tenant-scoped profile data returned by the repository layer.

    Returns:
        DurableProfile:
            Runtime durable profile with copied preference, fact, and event payloads.
    """
    return DurableProfile(
        customer_id=stored.customer_id,
        tenant_id=stored.tenant_id,
        preferences=dict(stored.preferences),
        dietary_facts=dict(stored.dietary_facts),
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