"""Tenant-scoped repositories for customer profiles and episodic memory.

This module is the database boundary for durable customer memory. Every public
function receives a tenant ID and customer ID, and all customer lookups verify
tenant ownership before profile, preference, dietary-fact, event, or deletion
operations occur.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cafe_assistant.db.models import Customer, CustomerProfile, EpisodicEvent


@dataclass(frozen=True, slots=True)
class StoredProfile:
    """Repository DTO for durable customer profile data.

    Attributes:
        customer_id (int):
            Durable customer ID within the tenant.
        tenant_id (int):
            Tenant that owns the profile.
        preferences (dict[str, object]):
            Auto-writable UI preferences stored for the customer.
        dietary_facts (dict[str, object]):
            Consent-gated dietary/health facts stored for the customer.
        consent_at (datetime | None):
            Time when dietary/health consent was granted, when present.
        recent_events (list[EpisodicEvent]):
            Recent episodic event ORM rows for inspection.
    """

    customer_id: int
    tenant_id: int
    preferences: dict[str, object]
    dietary_facts: dict[str, object]
    consent_at: datetime | None
    recent_events: list[EpisodicEvent]


async def get_customer(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
) -> Customer | None:
    """Load a customer only when it belongs to the requested tenant.

    Args:
        session (AsyncSession):
            Async database session used for the lookup.
        tenant_id (int):
            Tenant scope that must match the customer row.
        customer_id (int):
            Durable customer ID to load.

    Returns:
        Customer | None:
            Customer with profile eagerly loaded, or None on tenant mismatch/miss.
    """
    return await session.scalar(
        select(Customer)
        .where(Customer.id == customer_id, Customer.tenant_id == tenant_id)
        .options(selectinload(Customer.profile))
    )


async def load_stored_profile(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
    event_limit: int = 10,
) -> StoredProfile | None:
    """Load stored profile data and recent events for one tenant/customer.

    Args:
        session (AsyncSession):
            Async database session used for profile and event reads.
        tenant_id (int):
            Tenant that must own the customer.
        customer_id (int):
            Durable customer ID within the tenant.
        event_limit (int):
            Maximum number of recent episodic events to return.

    Returns:
        StoredProfile | None:
            Stored profile DTO when the tenant/customer pair exists, otherwise None.
    """
    customer = await get_customer(session, tenant_id=tenant_id, customer_id=customer_id)
    if customer is None:
        return None
    profile = await ensure_customer_profile(session, customer)
    events = list(
        await session.scalars(
            select(EpisodicEvent)
            .where(EpisodicEvent.customer_id == customer_id)
            .order_by(EpisodicEvent.created_at.desc(), EpisodicEvent.id.desc())
            .limit(event_limit)
        )
    )
    events.reverse()
    return StoredProfile(
        customer_id=customer.id,
        tenant_id=customer.tenant_id,
        preferences=dict(profile.preferences or {}),
        dietary_facts=dict(profile.dietary_facts or {}),
        consent_at=profile.consent_at,
        recent_events=events,
    )


async def ensure_customer_profile(
    session: AsyncSession,
    customer: Customer,
) -> CustomerProfile:
    """Ensure a customer has a profile row.

    Args:
        session (AsyncSession):
            Async database session used to add a missing profile.
        customer (Customer):
            Tenant-scoped customer ORM object.

    Returns:
        CustomerProfile:
            Existing or newly-created profile row for the customer.
    """
    if customer.profile is not None:
        return customer.profile
    profile = CustomerProfile(customer_id=customer.id, preferences={}, dietary_facts={})
    session.add(profile)
    await session.flush()
    return profile


async def update_preferences(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
    updates: dict[str, object],
) -> bool:
    """Merge auto-writable preference updates into a tenant-scoped profile.

    Args:
        session (AsyncSession):
            Async database session used for the update.
        tenant_id (int):
            Tenant that must own the customer profile.
        customer_id (int):
            Durable customer ID within the tenant.
        updates (dict[str, object]):
            Preference keys and values to merge into profile JSON.

    Returns:
        bool:
            True when the profile was found and updated, otherwise False.
    """
    customer = await get_customer(session, tenant_id=tenant_id, customer_id=customer_id)
    if customer is None:
        return False
    profile = await ensure_customer_profile(session, customer)
    profile.preferences = {**dict(profile.preferences or {}), **updates}
    await session.flush()
    return True


async def update_dietary_facts(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
    updates: dict[str, object],
) -> bool:
    """Merge consent-gated dietary facts into a tenant-scoped profile.

    Args:
        session (AsyncSession):
            Async database session used for the update.
        tenant_id (int):
            Tenant that must own the customer profile.
        customer_id (int):
            Durable customer ID within the tenant.
        updates (dict[str, object]):
            Dietary/health fact keys and values approved by the write gate.

    Returns:
        bool:
            True when the profile was found and updated, otherwise False.
    """
    customer = await get_customer(session, tenant_id=tenant_id, customer_id=customer_id)
    if customer is None:
        return False
    profile = await ensure_customer_profile(session, customer)
    profile.dietary_facts = {**dict(profile.dietary_facts or {}), **updates}
    profile.consent_at = _utcnow()
    await session.flush()
    return True


async def append_event(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
    event_type: str,
    payload: dict[str, object],
) -> bool:
    """Append an episodic event for a tenant-scoped customer.

    Args:
        session (AsyncSession):
            Async database session used to write the event.
        tenant_id (int):
            Tenant that must own the customer.
        customer_id (int):
            Durable customer ID within the tenant.
        event_type (str):
            Short event category string.
        payload (dict[str, object]):
            Minimal structured event metadata.

    Returns:
        bool:
            True when the event was written, otherwise False on tenant/customer miss.
    """
    customer = await get_customer(session, tenant_id=tenant_id, customer_id=customer_id)
    if customer is None:
        return False
    session.add(
        EpisodicEvent(
            customer_id=customer_id,
            type=event_type,
            payload=payload,
        )
    )
    await session.flush()
    return True


async def delete_customer_profile(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
) -> bool:
    """Delete a tenant-scoped customer and cascaded memory rows.

    Args:
        session (AsyncSession):
            Async database session used for deletion.
        tenant_id (int):
            Tenant that must own the customer.
        customer_id (int):
            Durable customer ID within the tenant.

    Returns:
        bool:
            True when a customer was found and deleted, otherwise False.
    """
    customer = await get_customer(session, tenant_id=tenant_id, customer_id=customer_id)
    if customer is None:
        return False
    await session.delete(customer)
    await session.flush()
    return True


def _utcnow() -> datetime:
    """Return the current timezone-aware UTC timestamp.

    Args:
        None:
            This helper has no inputs.

    Returns:
        datetime:
            Current UTC timestamp.
    """
    return datetime.now(UTC)