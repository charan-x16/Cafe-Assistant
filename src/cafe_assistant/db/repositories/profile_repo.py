from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cafe_assistant.db.models import Customer, CustomerProfile, EpisodicEvent


@dataclass(frozen=True, slots=True)
class StoredProfile:
    customer_id: int
    tenant_id: int
    preferences: dict[str, object]
    dietary_facts: dict[str, object]
    consent_at: datetime | None
    recent_events: list[EpisodicEvent]


async def get_or_create_customer_by_phone(
    session: AsyncSession,
    *,
    tenant_id: int,
    phone_hash: str,
) -> Customer:
    customer = await session.scalar(
        select(Customer)
        .where(Customer.tenant_id == tenant_id, Customer.phone_hash == phone_hash)
        .options(selectinload(Customer.profile))
    )
    if customer is None:
        customer = Customer(tenant_id=tenant_id, phone_hash=phone_hash)
        customer.profile = CustomerProfile(preferences={}, dietary_facts={})
        session.add(customer)
        await session.flush()
        return customer

    if customer.profile is None:
        customer.profile = CustomerProfile(preferences={}, dietary_facts={})
        await session.flush()
    return customer


async def get_customer(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
) -> Customer | None:
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
    customer = await get_customer(session, tenant_id=tenant_id, customer_id=customer_id)
    if customer is None:
        return False
    await session.delete(customer)
    await session.flush()
    return True


def _utcnow() -> datetime:
    return datetime.now(UTC)
