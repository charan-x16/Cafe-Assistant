"""Tenant-scoped consent repository for durable health-data writes.

Consent records authorize specific durable-memory scopes for a customer inside a
tenant. The write gate calls this module before saving allergies, dietary modes,
or diabetes-related low-sugar facts. All public operations verify the customer
belongs to the supplied tenant before reading or mutating consent rows.
"""

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
    """Grant a consent scope for a tenant-scoped customer.

    Args:
        session (AsyncSession):
            Async database session used for consent/profile writes.
        tenant_id (int):
            Tenant that must own the customer.
        customer_id (int):
            Durable customer ID within the tenant.
        scope (str):
            Consent scope to grant.

    Returns:
        bool:
            True when the customer exists and consent is active or newly granted.
    """
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
    """Check whether a tenant-scoped customer has active consent.

    Args:
        session (AsyncSession):
            Async database session used for lookup.
        tenant_id (int):
            Tenant that must own the customer.
        customer_id (int):
            Durable customer ID within the tenant.
        scope (str):
            Consent scope being checked.

    Returns:
        bool:
            True only when the customer belongs to the tenant and has non-revoked consent.
    """
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
    """Revoke all active consent scopes for a tenant-scoped customer.

    Args:
        session (AsyncSession):
            Async database session used for updates.
        tenant_id (int):
            Tenant that must own the customer.
        customer_id (int):
            Durable customer ID within the tenant.

    Returns:
        bool:
            True when the customer exists and revocation timestamps were applied.
    """
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
    """Load one active consent row by customer and scope.

    Args:
        session (AsyncSession):
            Async database session used for lookup.
        customer_id (int):
            Durable customer ID already verified by the caller.
        scope (str):
            Consent scope to find.

    Returns:
        Consent | None:
            Non-revoked consent row when present, otherwise None.
    """
    return await session.scalar(
        select(Consent).where(
            Consent.customer_id == customer_id,
            Consent.scope == scope,
            Consent.revoked_at.is_(None),
        )
    )


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