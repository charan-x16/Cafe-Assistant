from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.models import Consent
from cafe_assistant.db.repositories.profile_repo import ensure_customer_profile, get_customer

DIETARY_HEALTH_SCOPE = "dietary_health"


async def grant_consent(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
    scope: str,
) -> bool:
    customer = await get_customer(session, tenant_id=tenant_id, customer_id=customer_id)
    if customer is None:
        return False

    active = await _active_consent(session, customer_id=customer_id, scope=scope)
    if active is None:
        session.add(Consent(customer_id=customer_id, scope=scope))
    if scope == DIETARY_HEALTH_SCOPE:
        profile = await ensure_customer_profile(session, customer)
        profile.consent_at = _utcnow()
    await session.flush()
    return True


async def has_active_consent(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
    scope: str,
) -> bool:
    customer = await get_customer(session, tenant_id=tenant_id, customer_id=customer_id)
    if customer is None:
        return False
    return await _active_consent(session, customer_id=customer_id, scope=scope) is not None


async def revoke_all_consents(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
) -> bool:
    customer = await get_customer(session, tenant_id=tenant_id, customer_id=customer_id)
    if customer is None:
        return False

    consents = await session.scalars(
        select(Consent).where(Consent.customer_id == customer_id, Consent.revoked_at.is_(None))
    )
    revoked_at = _utcnow()
    for consent in consents:
        consent.revoked_at = revoked_at
    await session.flush()
    return True


async def _active_consent(
    session: AsyncSession,
    *,
    customer_id: int,
    scope: str,
) -> Consent | None:
    return await session.scalar(
        select(Consent).where(
            Consent.customer_id == customer_id,
            Consent.scope == scope,
            Consent.revoked_at.is_(None),
        )
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)
