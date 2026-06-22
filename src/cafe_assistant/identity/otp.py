"""Phone OTP consent-upgrade service for durable customer identity.

The OTP flow is the only path that links an anonymous session to a durable
customer profile with health-data consent. Challenges store hashed phone/code
values, expire through their backing store, validate tenant ownership, and grant
only allowlisted consent scopes. Tests inject deterministic senders and in-memory
stores; production should use the Redis store plus a real SMS sender adapter.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.config import settings
from cafe_assistant.db.repositories.consent_repo import (
    DIETARY_HEALTH_SCOPE,
    grant_consent,
    validate_consent_scopes,
)
from cafe_assistant.db.repositories.profile_repo import (
    append_event,
    get_or_create_customer_by_phone,
    update_dietary_facts,
)
from cafe_assistant.domain.dietary import CustomerRestrictions
from cafe_assistant.identity.device import issue_device_token

if TYPE_CHECKING:
    from cafe_assistant.memory.session import SessionState

_GENERIC_OTP_CONFIRM_ERROR = "OTP challenge could not be confirmed."
_LOCAL_ENVIRONMENTS = frozenset({"local", "test", "development"})
_default_otp_service: OtpService | None = None


class OtpError(ValueError):
    """Raised when an OTP challenge cannot be started or confirmed."""


class SmsSender(Protocol):
    """Interface for SMS adapters used by the OTP service."""

    async def send_otp(self, phone: str, code: str) -> None:
        """Send a one-time code to a customer phone number."""


class NoopSmsSender:
    """Development-only SMS sender that intentionally delivers nothing."""

    def __init__(self, *, allow_send: bool | None = None) -> None:
        """Create a no-op sender with an explicit safety switch.

        Args:
            allow_send (bool | None):
                Whether no-op sends are allowed. None uses local/test environment rules.

        Returns:
            None:
                The sender is ready for dependency injection.
        """
        self.allow_send = _allow_noop_sms() if allow_send is None else allow_send

    async def send_otp(self, phone: str, code: str) -> None:
        """Accept or reject a no-op OTP delivery attempt.

        Args:
            phone (str):
                Normalized destination phone number.
            code (str):
                Generated one-time code, intentionally discarded by this sender.

        Returns:
            None:
                The call succeeds only when no-op sending is allowed for this environment.
        """
        del phone, code
        if not self.allow_send:
            raise OtpError("OTP SMS sender is not configured.")


@dataclass(frozen=True, slots=True)
class OtpStartResult:
    """Result returned after creating an OTP challenge.

    Attributes:
        challenge_id (str):
            Opaque challenge identifier returned to the browser.
        expires_at (datetime):
            Time after which the challenge cannot be confirmed.
    """

    challenge_id: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class OtpConfirmResult:
    """Result returned after a successful OTP confirmation.

    Attributes:
        customer_id (int):
            Durable customer ID created or loaded for the normalized phone hash.
        tenant_id (int):
            Tenant that owns the linked customer profile.
        device_token (str):
            New opaque browser token issued for future recognition.
        granted_scopes (tuple[str, ...]):
            Consent scopes granted by the confirmed challenge.
    """

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
    attempts: int = 0


class OtpStore(Protocol):
    """Storage interface for short-lived OTP challenges."""

    async def save(self, challenge_id: str, challenge: _OtpChallenge) -> None:
        """Persist or replace an OTP challenge until its expiration time."""

    async def load(self, challenge_id: str) -> _OtpChallenge | None:
        """Load an OTP challenge without consuming it."""

    async def delete(self, challenge_id: str) -> None:
        """Delete an OTP challenge after success, expiry, or too many failures."""


class InMemoryOtpStore:
    """In-process OTP store for tests and local single-worker development."""

    def __init__(self) -> None:
        """Initialize an empty challenge map.

        Args:
            None:
                The store has no external dependencies.

        Returns:
            None:
                The store is ready to accept challenges.
        """
        self._challenges: dict[str, _OtpChallenge] = {}

    async def save(self, challenge_id: str, challenge: _OtpChallenge) -> None:
        """Save a challenge in memory.

        Args:
            challenge_id (str):
                Opaque challenge ID.
            challenge (_OtpChallenge):
                Challenge data containing hashes, tenant, scopes, and attempts.

        Returns:
            None:
                The challenge map is updated.
        """
        self._challenges[challenge_id] = challenge

    async def load(self, challenge_id: str) -> _OtpChallenge | None:
        """Load a challenge from memory without consuming it.

        Args:
            challenge_id (str):
                Opaque challenge ID to read.

        Returns:
            _OtpChallenge | None:
                Stored challenge, or None when missing.
        """
        return self._challenges.get(challenge_id)

    async def delete(self, challenge_id: str) -> None:
        """Delete a challenge from memory.

        Args:
            challenge_id (str):
                Opaque challenge ID to delete.

        Returns:
            None:
                Missing challenges are ignored.
        """
        self._challenges.pop(challenge_id, None)


class RedisOtpStore:
    """Redis-backed OTP store with per-challenge TTL expiry."""

    def __init__(self, redis: object, *, key_prefix: str = "cafe_assistant:otp") -> None:
        """Create a Redis OTP store.

        Args:
            redis (object):
                Async Redis client exposing get, set, and delete methods.
            key_prefix (str):
                Prefix used to isolate OTP keys from other Redis data.

        Returns:
            None:
                The store keeps the Redis client for later operations.
        """
        self.redis = redis
        self.key_prefix = key_prefix

    async def save(self, challenge_id: str, challenge: _OtpChallenge) -> None:
        """Save a challenge in Redis with TTL based on expiration time.

        Args:
            challenge_id (str):
                Opaque challenge ID.
            challenge (_OtpChallenge):
                Challenge payload to serialize.

        Returns:
            None:
                Redis receives the serialized challenge with an expiry.
        """
        ttl = max(1, int((challenge.expires_at - _utcnow()).total_seconds()))
        await self.redis.set(  # type: ignore[attr-defined]
            self._key(challenge_id),
            json.dumps(_challenge_to_payload(challenge)),
            ex=ttl,
        )

    async def load(self, challenge_id: str) -> _OtpChallenge | None:
        """Load and deserialize a challenge from Redis.

        Args:
            challenge_id (str):
                Opaque challenge ID to read.

        Returns:
            _OtpChallenge | None:
                Challenge when the Redis key exists and decodes, otherwise None.
        """
        payload = await self.redis.get(self._key(challenge_id))  # type: ignore[attr-defined]
        if payload is None:
            return None
        try:
            return _challenge_from_payload(json.loads(payload))
        except (TypeError, ValueError, json.JSONDecodeError):
            await self.delete(challenge_id)
            return None

    async def delete(self, challenge_id: str) -> None:
        """Delete a challenge from Redis.

        Args:
            challenge_id (str):
                Opaque challenge ID to delete.

        Returns:
            None:
                Redis delete is issued for the challenge key.
        """
        await self.redis.delete(self._key(challenge_id))  # type: ignore[attr-defined]

    def _key(self, challenge_id: str) -> str:
        """Build the Redis key for one challenge.

        Args:
            challenge_id (str):
                Opaque challenge ID.

        Returns:
            str:
                Namespaced Redis key.
        """
        return f"{self.key_prefix}:{challenge_id}"


class OtpService:
    """Service that starts and confirms OTP consent-upgrade challenges."""

    def __init__(
        self,
        *,
        sender: SmsSender | None = None,
        store: OtpStore | None = None,
    ) -> None:
        """Create an OTP service with injectable sender and store.

        Args:
            sender (SmsSender | None):
                SMS sender adapter. None uses the environment-aware no-op sender.
            store (OtpStore | None):
                Challenge store. None uses in-memory storage for tests/local development.

        Returns:
            None:
                The service stores dependencies for start and confirm calls.
        """
        self.sender = sender or NoopSmsSender()
        self.store = store or InMemoryOtpStore()

    async def start(
        self,
        *,
        tenant_id: int,
        phone: str,
        scopes: tuple[str, ...] = (DIETARY_HEALTH_SCOPE,),
    ) -> OtpStartResult:
        """Start an OTP challenge for allowed consent scopes.

        Args:
            tenant_id (int):
                Tenant that owns the future durable customer profile.
            phone (str):
                Customer phone number used only after normalization and hashing.
            scopes (tuple[str, ...]):
                Requested consent scopes, validated against the allowlist.

        Returns:
            OtpStartResult:
                Opaque challenge ID and expiry timestamp.
        """
        self._validate_runtime_configuration()
        validated_scopes = validate_consent_scopes(scopes)
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
                scopes=validated_scopes,
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
        """Confirm an OTP challenge and link durable identity.

        Args:
            session (AsyncSession):
                Async database session used for customer, consent, profile, and token writes.
            tenant_id (int):
                Tenant that must match the original challenge.
            phone (str):
                Phone number that must match the original challenge hash.
            challenge_id (str):
                Opaque challenge ID returned by `start`.
            code (str):
                One-time code delivered by the SMS sender.
            session_state (SessionState | None):
                Tenant-scoped session memory whose current restrictions may be
                written only after dietary-health consent is granted.

        Returns:
            OtpConfirmResult:
                Linked customer identity, new device token, and granted scopes.
        """
        normalized_phone = normalize_phone(phone)
        challenge = await self.store.load(challenge_id)
        if challenge is None:
            raise OtpError(_GENERIC_OTP_CONFIRM_ERROR)
        if challenge.expires_at <= _utcnow():
            await self.store.delete(challenge_id)
            raise OtpError(_GENERIC_OTP_CONFIRM_ERROR)
        if challenge.tenant_id != tenant_id:
            await self._record_failed_attempt(challenge_id, challenge)
            raise OtpError(_GENERIC_OTP_CONFIRM_ERROR)
        if challenge.phone_hash != hash_phone(normalized_phone):
            await self._record_failed_attempt(challenge_id, challenge)
            raise OtpError(_GENERIC_OTP_CONFIRM_ERROR)
        if not hmac.compare_digest(challenge.code_hash, _hash_code(code)):
            await self._record_failed_attempt(challenge_id, challenge)
            raise OtpError(_GENERIC_OTP_CONFIRM_ERROR)

        await self.store.delete(challenge_id)
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
    async def delete_challenge(self, challenge_id: str) -> None:
        """Delete a pending OTP challenge when a data-rights request supplies it.

        Args:
            challenge_id (str):
                Opaque challenge ID previously returned by the OTP start endpoint.

        Returns:
            None:
                The underlying challenge store receives a delete; missing challenges are ignored.
        """
        await self.store.delete(challenge_id)

    async def _record_failed_attempt(
        self,
        challenge_id: str,
        challenge: _OtpChallenge,
    ) -> None:
        """Record a failed confirmation attempt and delete after max attempts.

        Args:
            challenge_id (str):
                Opaque challenge ID being attempted.
            challenge (_OtpChallenge):
                Current challenge state loaded from the store.

        Returns:
            None:
                The store is updated or the challenge is deleted.
        """
        attempts = challenge.attempts + 1
        if attempts >= settings.otp_max_attempts:
            await self.store.delete(challenge_id)
            return
        await self.store.save(
            challenge_id,
            _OtpChallenge(
                tenant_id=challenge.tenant_id,
                phone_hash=challenge.phone_hash,
                code_hash=challenge.code_hash,
                expires_at=challenge.expires_at,
                scopes=challenge.scopes,
                attempts=attempts,
            ),
        )

    def _validate_runtime_configuration(self) -> None:
        """Fail closed for production OTP runtime misconfiguration.

        Args:
            None:
                The check reads environment-backed settings and injected dependencies.

        Returns:
            None:
                The function returns only when the OTP service is safe to use.
        """
        environment = settings.environment.strip().lower()
        if environment in _LOCAL_ENVIRONMENTS:
            return
        if isinstance(self.sender, NoopSmsSender):
            raise OtpError("OTP SMS sender is not configured.")
        if isinstance(self.store, InMemoryOtpStore):
            raise OtpError("OTP store must be Redis-backed outside local/test environments.")


def normalize_phone(phone: str) -> str:
    """Normalize and validate a phone number for hashing and OTP delivery.

    Args:
        phone (str):
            Customer phone number supplied by the browser.

    Returns:
        str:
            Normalized phone string preserving a leading plus sign when present.
    """
    stripped = phone.strip()
    prefix = "+" if stripped.startswith("+") else ""
    digits = re.sub(r"[\s().-]+", "", stripped[1:] if prefix else stripped)
    if not digits.isdigit() or len(digits) < 7:
        raise OtpError("Phone number is invalid.")
    return f"{prefix}{digits}"


def hash_phone(phone: str) -> str:
    """Hash a normalized phone number for tenant-scoped identity lookup.

    Args:
        phone (str):
            Raw or normalized phone number.

    Returns:
        str:
            Hex HMAC digest computed with the phone-hash secret.
    """
    return hmac.new(
        settings.identity_phone_hash_secret.encode("utf-8"),
        normalize_phone(phone).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def restrictions_to_dietary_facts(
    restrictions: CustomerRestrictions,
) -> dict[str, object]:
    """Convert explicit restrictions into durable dietary fact JSON.

    Args:
        restrictions (CustomerRestrictions):
            Current-turn restrictions that were explicitly mentioned by the customer.

    Returns:
        dict[str, object]:
            JSON-serializable dietary facts suitable for consent-gated persistence.
    """
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


def get_default_otp_service() -> OtpService:
    """Build the default OTP service for API routes.

    Args:
        None:
            The service is configured from environment settings.

    Returns:
        OtpService:
            OTP service using Redis outside local/test or when explicitly configured.
    """
    global _default_otp_service
    if _default_otp_service is None:
        _default_otp_service = OtpService(sender=NoopSmsSender(), store=_default_otp_store())
    return _default_otp_service


def _default_otp_store() -> OtpStore:
    """Select the default OTP store from environment settings.

    Args:
        None:
            Reads the configured store provider and environment.

    Returns:
        OtpStore:
            Redis store for production/non-local settings, otherwise in-memory store.
    """
    provider = settings.otp_store_provider.strip().lower()
    environment = settings.environment.strip().lower()
    if provider == "redis" or environment not in _LOCAL_ENVIRONMENTS:
        from redis.asyncio import Redis

        return RedisOtpStore(Redis.from_url(settings.redis_url, decode_responses=True))
    return InMemoryOtpStore()


def _allow_noop_sms() -> bool:
    """Determine whether the no-op SMS sender may accept send requests.

    Args:
        None:
            Reads environment and explicit override settings.

    Returns:
        bool:
            True only for local/test environments or explicit non-production override.
    """
    environment = settings.environment.strip().lower()
    if environment in _LOCAL_ENVIRONMENTS:
        return True
    return settings.otp_allow_noop_sms_sender and environment not in {"production", "prod"}


def _generate_code() -> str:
    """Generate a six-digit one-time code.

    Args:
        None:
            The code is generated from the system CSPRNG.

    Returns:
        str:
            Zero-padded six-digit OTP code.
    """
    return f"{secrets.randbelow(1_000_000):06d}"


def _hash_code(code: str) -> str:
    """Hash an OTP code with the OTP-specific HMAC secret.

    Args:
        code (str):
            One-time code generated or supplied for confirmation.

    Returns:
        str:
            Hex HMAC digest used for constant-time comparison.
    """
    return hmac.new(
        settings.identity_otp_hash_secret.encode("utf-8"),
        code.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _challenge_to_payload(challenge: _OtpChallenge) -> dict[str, object]:
    """Serialize a challenge for Redis storage.

    Args:
        challenge (_OtpChallenge):
            In-memory challenge data.

    Returns:
        dict[str, object]:
            JSON-compatible challenge payload.
    """
    return {
        "tenant_id": challenge.tenant_id,
        "phone_hash": challenge.phone_hash,
        "code_hash": challenge.code_hash,
        "expires_at": challenge.expires_at.isoformat(),
        "scopes": list(challenge.scopes),
        "attempts": challenge.attempts,
    }


def _challenge_from_payload(payload: dict[str, object]) -> _OtpChallenge:
    """Deserialize a Redis payload into a challenge.

    Args:
        payload (dict[str, object]):
            JSON object loaded from Redis.

    Returns:
        _OtpChallenge:
            Validated challenge object.
    """
    return _OtpChallenge(
        tenant_id=int(payload["tenant_id"]),
        phone_hash=str(payload["phone_hash"]),
        code_hash=str(payload["code_hash"]),
        expires_at=datetime.fromisoformat(str(payload["expires_at"])),
        scopes=tuple(str(scope) for scope in payload.get("scopes", [])),
        attempts=int(payload.get("attempts", 0)),
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
