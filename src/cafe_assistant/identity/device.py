from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.config import settings
from cafe_assistant.db.models import Customer, CustomerDeviceToken
from cafe_assistant.db.repositories.profile_repo import get_customer


class TenantIsolationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    tenant_id: int
    customer_id: int


async def issue_device_token(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
) -> str:
    customer = await get_customer(session, tenant_id=tenant_id, customer_id=customer_id)
    if customer is None:
        raise TenantIsolationError("Customer does not belong to the tenant.")

    token = secrets.token_urlsafe(settings.device_token_bytes)
    session.add(
        CustomerDeviceToken(
            tenant_id=tenant_id,
            customer_id=customer_id,
            token_hash=_hash_token(token),
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
    if not token:
        return None

    record = await session.scalar(
        select(CustomerDeviceToken)
        .join(Customer, Customer.id == CustomerDeviceToken.customer_id)
        .where(
            CustomerDeviceToken.tenant_id == tenant_id,
            CustomerDeviceToken.token_hash == _hash_token(token),
            Customer.tenant_id == tenant_id,
        )
    )
    if record is None:
        return None

    record.last_seen_at = datetime.now(UTC)
    await session.flush()
    return DeviceIdentity(tenant_id=tenant_id, customer_id=record.customer_id)


def _hash_token(token: str) -> str:
    return hmac.new(
        settings.identity_phone_hash_secret.encode("utf-8"),
        token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
