from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.config import settings
from cafe_assistant.db.repositories.consent_repo import DIETARY_HEALTH_SCOPE, grant_consent
from cafe_assistant.db.repositories.profile_repo import (
    append_event,
    get_or_create_customer_by_phone,
    update_dietary_facts,
)
from cafe_assistant.domain.dietary import CustomerRestrictions
from cafe_assistant.identity.device import issue_device_token
from cafe_assistant.memory.session import SessionState


class OtpError(ValueError):
    pass


class SmsSender(Protocol):
    async def send_otp(self, phone: str, code: str) -> None:
        """Send a one-time code to a customer phone number."""


class NoopSmsSender:
    async def send_otp(self, phone: str, code: str) -> None:
        del phone, code


@dataclass(frozen=True, slots=True)
class OtpStartResult:
    challenge_id: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class OtpConfirmResult:
    customer_id: int
    tenant_id: int
    device_token: str
    granted_scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _OtpChallenge:
    tenant_id: int
    phone_hash: str
    code_hash: str
    expires_at: datetime
    scopes: tuple[str, ...]


class InMemoryOtpStore:
    def __init__(self) -> None:
        self._challenges: dict[str, _OtpChallenge] = {}

    async def save(self, challenge_id: str, challenge: _OtpChallenge) -> None:
        self._challenges[challenge_id] = challenge

    async def pop(self, challenge_id: str) -> _OtpChallenge | None:
        return self._challenges.pop(challenge_id, None)


class OtpService:
    def __init__(
        self,
        *,
        sender: SmsSender | None = None,
        store: InMemoryOtpStore | None = None,
    ) -> None:
        self.sender = sender or NoopSmsSender()
        self.store = store or InMemoryOtpStore()

    async def start(
        self,
        *,
        tenant_id: int,
        phone: str,
        scopes: tuple[str, ...] = (DIETARY_HEALTH_SCOPE,),
    ) -> OtpStartResult:
        normalized_phone = normalize_phone(phone)
        now = _utcnow()
        code = _generate_code()
        challenge_id = secrets.token_urlsafe(18)
        expires_at = now + timedelta(seconds=settings.otp_code_ttl_seconds)
        await self.store.save(
            challenge_id,
            _OtpChallenge(
                tenant_id=tenant_id,
                phone_hash=hash_phone(normalized_phone),
                code_hash=_hash_code(code),
                expires_at=expires_at,
                scopes=tuple(sorted(set(scopes))),
            ),
        )
        await self.sender.send_otp(normalized_phone, code)
        return OtpStartResult(challenge_id=challenge_id, expires_at=expires_at)

    async def confirm(
        self,
        session: AsyncSession,
        *,
        tenant_id: int,
        phone: str,
        challenge_id: str,
        code: str,
        session_state: SessionState | None = None,
    ) -> OtpConfirmResult:
        normalized_phone = normalize_phone(phone)
        challenge = await self.store.pop(challenge_id)
        if challenge is None:
            raise OtpError("OTP challenge was not found or already used.")
        if challenge.tenant_id != tenant_id:
            raise OtpError("OTP challenge does not belong to this tenant.")
        if challenge.expires_at <= _utcnow():
            raise OtpError("OTP challenge expired.")
        if challenge.phone_hash != hash_phone(normalized_phone):
            raise OtpError("OTP phone number does not match the challenge.")
        if not hmac.compare_digest(challenge.code_hash, _hash_code(code)):
            raise OtpError("OTP code is invalid.")

        customer = await get_or_create_customer_by_phone(
            session,
            tenant_id=tenant_id,
            phone_hash=challenge.phone_hash,
        )
        granted_scopes = challenge.scopes
        for scope in granted_scopes:
            await grant_consent(
                session,
                tenant_id=tenant_id,
                customer_id=customer.id,
                scope=scope,
            )

        if session_state is not None and DIETARY_HEALTH_SCOPE in granted_scopes:
            facts = restrictions_to_dietary_facts(session_state.restrictions)
            if facts:
                await update_dietary_facts(
                    session,
                    tenant_id=tenant_id,
                    customer_id=customer.id,
                    updates=facts,
                )

        await append_event(
            session,
            tenant_id=tenant_id,
            customer_id=customer.id,
            event_type="otp_upgrade",
            payload={"granted_scopes": list(granted_scopes)},
        )
        device_token = await issue_device_token(
            session,
            tenant_id=tenant_id,
            customer_id=customer.id,
        )
        await session.flush()
        return OtpConfirmResult(
            customer_id=customer.id,
            tenant_id=tenant_id,
            device_token=device_token,
            granted_scopes=granted_scopes,
        )


def normalize_phone(phone: str) -> str:
    normalized = phone.strip().replace(" ", "").replace("-", "")
    if normalized.startswith("+"):
        digits = normalized[1:]
        prefix = "+"
    else:
        digits = normalized
        prefix = ""
    if not digits.isdigit() or len(digits) < 7:
        raise OtpError("Phone number is invalid.")
    return f"{prefix}{digits}"


def hash_phone(phone: str) -> str:
    return hmac.new(
        settings.identity_phone_hash_secret.encode("utf-8"),
        normalize_phone(phone).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def restrictions_to_dietary_facts(
    restrictions: CustomerRestrictions,
) -> dict[str, object]:
    facts: dict[str, object] = {}
    if restrictions.avoid_allergens:
        facts["avoid_allergens"] = sorted(
            allergen.value for allergen in restrictions.avoid_allergens
        )
    if restrictions.modes:
        facts["modes"] = sorted(mode.value for mode in restrictions.modes)
    if restrictions.prefer_low_sugar:
        facts["prefer_low_sugar"] = True
    return facts


def _generate_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def _hash_code(code: str) -> str:
    return hmac.new(
        settings.identity_phone_hash_secret.encode("utf-8"),
        code.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(UTC)
