from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from cafe_assistant.config import settings


class RateLimitExceededError(RuntimeError):
    def __init__(self, *, retry_after_seconds: int) -> None:
        super().__init__("Rate limit exceeded.")
        self.retry_after_seconds = retry_after_seconds


class RateLimiter(Protocol):
    async def check(
        self,
        *,
        session_id: str,
        client_ip: str,
    ) -> None:
        """Raise RateLimitExceededError when a limit is exceeded."""


@dataclass(frozen=True, slots=True)
class RateLimitRule:
    prefix: str
    limit: int
    window_seconds: int


class InMemoryRateLimiter:
    def __init__(
        self,
        *,
        session_limit: int = settings.rate_limit_session_requests,
        session_window_seconds: int = settings.rate_limit_session_window_seconds,
        ip_limit: int = settings.rate_limit_ip_requests,
        ip_window_seconds: int = settings.rate_limit_ip_window_seconds,
    ) -> None:
        self.rules = (
            RateLimitRule("session", session_limit, session_window_seconds),
            RateLimitRule("ip", ip_limit, ip_window_seconds),
        )
        self._buckets: dict[str, tuple[int, float]] = {}

    async def check(self, *, session_id: str, client_ip: str) -> None:
        now = time.time()
        for rule, identity in (
            (self.rules[0], session_id),
            (self.rules[1], client_ip),
        ):
            count, expires_at = self._buckets.get(
                _key(rule, identity),
                (0, now + rule.window_seconds),
            )
            if now >= expires_at:
                count = 0
                expires_at = now + rule.window_seconds
            count += 1
            self._buckets[_key(rule, identity)] = (count, expires_at)
            if count > rule.limit:
                raise RateLimitExceededError(
                    retry_after_seconds=max(1, int(expires_at - now)),
                )


class RedisRateLimiter:
    def __init__(
        self,
        redis: object,
        *,
        session_limit: int = settings.rate_limit_session_requests,
        session_window_seconds: int = settings.rate_limit_session_window_seconds,
        ip_limit: int = settings.rate_limit_ip_requests,
        ip_window_seconds: int = settings.rate_limit_ip_window_seconds,
    ) -> None:
        self.redis = redis
        self.rules = (
            RateLimitRule("session", session_limit, session_window_seconds),
            RateLimitRule("ip", ip_limit, ip_window_seconds),
        )

    async def check(self, *, session_id: str, client_ip: str) -> None:
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
    from redis.asyncio import Redis

    return RedisRateLimiter(Redis.from_url(settings.redis_url, decode_responses=True))


def _key(rule: RateLimitRule, identity: str) -> str:
    return f"cafe_assistant:rate:{rule.prefix}:{identity}"
