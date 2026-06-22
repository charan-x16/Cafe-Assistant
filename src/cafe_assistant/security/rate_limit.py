"""Tenant-aware request throttling for public API routes.

The API dependency passes tenant-scoped session and IP identities into this module.
Both the in-memory and Redis implementations hash those identities before using
storage keys, so browser session IDs and client IP addresses do not become raw
operational data in Redis, logs, or test diagnostics. The Redis implementation is
used in production; the in-memory implementation keeps integration tests fast and
deterministic.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Protocol

from cafe_assistant.config import settings


class RateLimitExceededError(RuntimeError):
    """Raised when a rate-limit bucket exceeds its configured request count."""

    def __init__(self, *, retry_after_seconds: int) -> None:
        """Create an error carrying the retry window for HTTP responses.

        Args:
            retry_after_seconds (int):
                Number of seconds clients should wait before retrying.

        Returns:
            None:
                The exception is initialized with a stable message and retry value.
        """
        super().__init__("Rate limit exceeded.")
        self.retry_after_seconds = retry_after_seconds


class RateLimiter(Protocol):
    """Storage-agnostic interface used by FastAPI rate-limit dependencies."""

    async def check(
        self,
        *,
        session_id: str,
        client_ip: str,
    ) -> None:
        """Raise when either tenant-scoped identity exceeds its limit.

        Args:
            session_id (str):
                Tenant-scoped session identity built by the API dependency.
            client_ip (str):
                Tenant-scoped IP identity built by the API dependency.

        Returns:
            None:
                Implementations return only when the request is allowed.
        """


@dataclass(frozen=True, slots=True)
class RateLimitRule:
    """Configuration for one fixed-window rate-limit bucket.

    Attributes:
        prefix (str):
            Bucket family, such as `session` or `ip`.
        limit (int):
            Maximum allowed requests during the window.
        window_seconds (int):
            Fixed window length in seconds.
    """

    prefix: str
    limit: int
    window_seconds: int


class InMemoryRateLimiter:
    """Deterministic in-process limiter used by tests and local development."""

    def __init__(
        self,
        *,
        session_limit: int = settings.rate_limit_session_requests,
        session_window_seconds: int = settings.rate_limit_session_window_seconds,
        ip_limit: int = settings.rate_limit_ip_requests,
        ip_window_seconds: int = settings.rate_limit_ip_window_seconds,
    ) -> None:
        """Create fixed-window buckets for session and IP identities.

        Args:
            session_limit (int):
                Request count allowed per session window.
            session_window_seconds (int):
                Session limit window length in seconds.
            ip_limit (int):
                Request count allowed per tenant-scoped IP window.
            ip_window_seconds (int):
                IP limit window length in seconds.

        Returns:
            None:
                The limiter stores empty buckets until requests arrive.
        """
        self.rules = (
            RateLimitRule("session", session_limit, session_window_seconds),
            RateLimitRule("ip", ip_limit, ip_window_seconds),
        )
        self._buckets: dict[str, tuple[int, float]] = {}

    async def check(self, *, session_id: str, client_ip: str) -> None:
        """Increment in-memory buckets and reject over-limit requests.

        Args:
            session_id (str):
                Tenant-scoped session identity.
            client_ip (str):
                Tenant-scoped IP identity.

        Returns:
            None:
                The request is allowed unless a bucket exceeds its configured limit.
        """
        now = time.time()
        for rule, identity in (
            (self.rules[0], session_id),
            (self.rules[1], client_ip),
        ):
            key = _key(rule, identity)
            count, expires_at = self._buckets.get(key, (0, now + rule.window_seconds))
            if now >= expires_at:
                count = 0
                expires_at = now + rule.window_seconds
            count += 1
            self._buckets[key] = (count, expires_at)
            if count > rule.limit:
                raise RateLimitExceededError(
                    retry_after_seconds=max(1, int(expires_at - now)),
                )


class RedisRateLimiter:
    """Redis-backed fixed-window limiter for deployed API instances."""

    def __init__(
        self,
        redis: object,
        *,
        session_limit: int = settings.rate_limit_session_requests,
        session_window_seconds: int = settings.rate_limit_session_window_seconds,
        ip_limit: int = settings.rate_limit_ip_requests,
        ip_window_seconds: int = settings.rate_limit_ip_window_seconds,
    ) -> None:
        """Create a Redis limiter using application rate-limit settings.

        Args:
            redis (object):
                Async Redis client exposing `incr`, `expire`, and `ttl`.
            session_limit (int):
                Request count allowed per session window.
            session_window_seconds (int):
                Session limit window length in seconds.
            ip_limit (int):
                Request count allowed per tenant-scoped IP window.
            ip_window_seconds (int):
                IP limit window length in seconds.

        Returns:
            None:
                The limiter stores the Redis client and bucket rules.
        """
        self.redis = redis
        self.rules = (
            RateLimitRule("session", session_limit, session_window_seconds),
            RateLimitRule("ip", ip_limit, ip_window_seconds),
        )

    async def check(self, *, session_id: str, client_ip: str) -> None:
        """Increment Redis buckets and reject over-limit requests.

        Args:
            session_id (str):
                Tenant-scoped session identity.
            client_ip (str):
                Tenant-scoped IP identity.

        Returns:
            None:
                The request is allowed unless Redis reports an over-limit bucket.
        """
        for rule, identity in (
            (self.rules[0], session_id),
            (self.rules[1], client_ip),
        ):
            key = _key(rule, identity)
            count = await self.redis.incr(key)  # type: ignore[attr-defined]
            if count == 1:
                await self.redis.expire(key, rule.window_seconds)  # type: ignore[attr-defined]
            if count > rule.limit:
                ttl = await self.redis.ttl(key)  # type: ignore[attr-defined]
                raise RateLimitExceededError(retry_after_seconds=max(1, int(ttl)))


def get_redis_rate_limiter() -> RedisRateLimiter:
    """Build the default Redis-backed limiter from environment settings.

    Args:
        None:
            Redis connection details are read from application settings.

    Returns:
        RedisRateLimiter:
            Limiter connected to the configured Redis URL.
    """
    from redis.asyncio import Redis

    return RedisRateLimiter(Redis.from_url(settings.redis_url, decode_responses=True))


def _key(rule: RateLimitRule, identity: str) -> str:
    """Build a Redis-safe key for one rate-limit identity.

    Args:
        rule (RateLimitRule):
            Rate-limit rule identifying the bucket family.
        identity (str):
            Raw tenant-scoped identity from API dependencies.

    Returns:
        str:
            Redis key containing only the bucket prefix and a HMAC digest.
    """
    return f"cafe_assistant:rate:{rule.prefix}:{_hash_identity(identity)}"


def _hash_identity(identity: str) -> str:
    """Hash a rate-limit identity before it reaches storage.

    Args:
        identity (str):
            Tenant-scoped session or IP identity.

    Returns:
        str:
            Hex HMAC digest derived with the configured rate-limit hash secret.
    """
    secret = settings.rate_limit_hash_secret or settings.identity_device_token_hash_secret
    return hmac.new(secret.encode("utf-8"), identity.encode("utf-8"), hashlib.sha256).hexdigest()
