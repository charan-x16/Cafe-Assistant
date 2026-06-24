"""Opaque tenant-scoped device-token issuing, verification, and revocation.

Device tokens are browser-held credentials created after a customer registers or
logs in. The raw token is returned once to the client; the server stores only an
HMAC hash scoped to a tenant/customer pair. Verification always
joins through the customer row so a token created for one tenant cannot recognize
a customer in another tenant.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.config import settings
from cafe_assistant.db.models import Customer, CustomerDeviceToken
from cafe_assistant.db.repositories.profile_repo import get_customer


class TenantIsolationError(ValueError):
    """Raised when a device-token operation crosses tenant ownership."""


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    """Verified durable identity represented by a browser device token.

    Attributes:
        tenant_id (int):
            Tenant that owns the customer and token.
        customer_id (int):
            Durable customer ID linked to the verified token.
    """

    tenant_id: int
    customer_id: int


async def issue_device_token(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
) -> str:
    """Issue a new opaque device token for a tenant-scoped customer.

    Args:
        session (AsyncSession):
            Async database session used for customer validation and token insert.
        tenant_id (int):
            Tenant that must own the customer.
        customer_id (int):
            Durable customer ID receiving the token.

    Returns:
        str:
            Raw opaque token to return to the browser exactly once.
    """
    customer = await get_customer(session, tenant_id=tenant_id, customer_id=customer_id)
    if customer is None:
        raise TenantIsolationError("Customer does not belong to the tenant.")

    token = secrets.token_urlsafe(settings.device_token_bytes)
    now = _utcnow()
    session.add(
        CustomerDeviceToken(
            tenant_id=tenant_id,
            customer_id=customer_id,
            token_hash=_hash_token(token),
            expires_at=now + timedelta(seconds=settings.device_token_ttl_seconds),
        )
    )
    await session.flush()
    return token


async def verify_device_token(
    session: AsyncSession,
    *,
    tenant_id: int,
    token: str | None,
) -> DeviceIdentity | None:
    """Verify an opaque token inside one tenant.

    Args:
        session (AsyncSession):
            Async database session used for lookup and last-seen update.
        tenant_id (int):
            Tenant scope that must match both token and customer rows.
        token (str | None):
            Raw device token supplied through an approved API transport.

    Returns:
        DeviceIdentity | None:
            Verified identity when the token is active, unexpired, and tenant-scoped;
            otherwise None so callers can continue anonymously.
    """
    if not token:
        return None

    now = _utcnow()
    record = await session.scalar(
        select(CustomerDeviceToken)
        .join(Customer, Customer.id == CustomerDeviceToken.customer_id)
        .where(
            CustomerDeviceToken.tenant_id == tenant_id,
            CustomerDeviceToken.token_hash == _hash_token(token),
            CustomerDeviceToken.revoked_at.is_(None),
            CustomerDeviceToken.expires_at > now,
            Customer.tenant_id == tenant_id,
        )
    )
    if record is None:
        return None

    record.last_seen_at = now
    await session.flush()
    return DeviceIdentity(tenant_id=tenant_id, customer_id=record.customer_id)


async def revoke_device_token(
    session: AsyncSession,
    *,
    tenant_id: int,
    token: str,
) -> bool:
    """Revoke one active device token inside a tenant.

    Args:
        session (AsyncSession):
            Async database session used to find and update the token row.
        tenant_id (int):
            Tenant that must own the token.
        token (str):
            Raw device token to revoke.

    Returns:
        bool:
            True when an active token was found and revoked, otherwise False.
    """
    record = await session.scalar(
        select(CustomerDeviceToken).where(
            CustomerDeviceToken.tenant_id == tenant_id,
            CustomerDeviceToken.token_hash == _hash_token(token),
            CustomerDeviceToken.revoked_at.is_(None),
        )
    )
    if record is None:
        return False
    record.revoked_at = _utcnow()
    await session.flush()
    return True


async def revoke_customer_device_tokens(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
) -> bool:
    """Revoke every active token for a tenant-scoped customer.

    Args:
        session (AsyncSession):
            Async database session used for customer validation and token updates.
        tenant_id (int):
            Tenant that must own the customer and tokens.
        customer_id (int):
            Durable customer whose tokens should be revoked.

    Returns:
        bool:
            True when the customer exists, regardless of whether active tokens existed.
    """
    customer = await get_customer(session, tenant_id=tenant_id, customer_id=customer_id)
    if customer is None:
        return False
    records = await session.scalars(
        select(CustomerDeviceToken).where(
            CustomerDeviceToken.tenant_id == tenant_id,
            CustomerDeviceToken.customer_id == customer_id,
            CustomerDeviceToken.revoked_at.is_(None),
        )
    )
    revoked_at = _utcnow()
    for record in records:
        record.revoked_at = revoked_at
    await session.flush()
    return True


def _hash_token(token: str) -> str:
    """Hash a raw device token with the device-token HMAC secret.

    Args:
        token (str):
            Raw opaque token supplied by the browser.

    Returns:
        str:
            Hex HMAC digest stored and compared by the database layer.
    """
    return hmac.new(
        settings.identity_device_token_hash_secret.encode("utf-8"),
        token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


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
