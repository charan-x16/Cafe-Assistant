"""Session-scoped memory for anonymous and recognized cafe chat sessions.

This module owns short-lived conversational memory stored in Redis for
production and in an in-memory dictionary for tests. Session memory is explicitly
scoped by both tenant ID and session ID so a browser-generated session ID reused
at another cafe cannot leak restrictions, preferences, or recent turns across
tenants. Durable profile memory lives in the database; this layer is only the
session overlay used before and during a chat run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from cafe_assistant.config import settings
from cafe_assistant.domain.dietary import AllergenCode, CustomerRestrictions, DietaryMode

if TYPE_CHECKING:
    from redis.asyncio import Redis


@dataclass(slots=True)
class ConversationTurn:
    """One recent conversational turn stored in session memory.

    Attributes:
        role (str):
            Speaker role, usually `user` or `assistant`.
        content (str):
            Raw turn text. Callers must treat this as untrusted user/model text.
    """

    role: str
    content: str


@dataclass(slots=True)
class SessionState:
    """Short-lived tenant-scoped state for one browser session.

    Attributes:
        restrictions (CustomerRestrictions):
            Active session restrictions learned during the anonymous or current
            visit. These are merged with durable profile restrictions at runtime.
        preferences (dict[str, object]):
            Non-health UI preferences, such as milk preference, remembered only
            for this session unless a recognized profile auto-writes them.
        recent_turns (list[ConversationTurn]):
            Recent conversation snippets retained for session continuity.
    """

    restrictions: CustomerRestrictions = field(
        default_factory=lambda: CustomerRestrictions(
            avoid_allergens=set(),
            modes=set(),
            prefer_low_sugar=False,
        )
    )
    preferences: dict[str, object] = field(default_factory=dict)
    recent_turns: list[ConversationTurn] = field(default_factory=list)


class SessionMemory(Protocol):
    """Protocol implemented by tenant-scoped session memory backends."""

    async def load(self, *, tenant_id: int, session_id: str) -> SessionState:
        """Load memory for one tenant/session pair.

        Args:
            tenant_id (int):
                Tenant that owns the session memory.
            session_id (str):
                Browser or client session identifier within the tenant.

        Returns:
            SessionState:
                Stored session state, or an empty state when no record exists.
        """

    async def save(self, *, tenant_id: int, session_id: str, state: SessionState) -> None:
        """Persist memory for one tenant/session pair.

        Args:
            tenant_id (int):
                Tenant that owns the session memory.
            session_id (str):
                Browser or client session identifier within the tenant.
            state (SessionState):
                Complete session state to serialize and store.

        Returns:
            None:
                Implementations complete through storage side effects.
        """
    async def delete(self, *, tenant_id: int, session_id: str) -> None:
        """Delete memory for one tenant/session pair.

        Args:
            tenant_id (int):
                Tenant that owns the session memory.
            session_id (str):
                Browser or client session identifier within the tenant.

        Returns:
            None:
                Missing records are ignored.
        """

class InMemorySessionMemory:
    """In-process tenant-scoped session memory used by tests."""

    def __init__(self) -> None:
        """Create an empty in-memory tenant/session state store.

        Args:
            None:
                The test store has no external dependencies.

        Returns:
            None:
                The backing dictionary is initialized empty.
        """
        self._states: dict[tuple[int, str], SessionState] = {}

    async def load(self, *, tenant_id: int, session_id: str) -> SessionState:
        """Load a copy of state for one tenant/session pair.

        Args:
            tenant_id (int):
                Tenant that owns the requested session.
            session_id (str):
                Session identifier within the tenant.

        Returns:
            SessionState:
                Defensive copy of stored state, or an empty state when missing.
        """
        return _copy_state(self._states.get((tenant_id, session_id), SessionState()))

    async def save(self, *, tenant_id: int, session_id: str, state: SessionState) -> None:
        """Save a defensive copy of state for one tenant/session pair.

        Args:
            tenant_id (int):
                Tenant that owns the session being saved.
            session_id (str):
                Session identifier within the tenant.
            state (SessionState):
                State to copy into the test store.

        Returns:
            None:
                The state is stored in memory under a tenant/session tuple key.
        """
        self._states[(tenant_id, session_id)] = _copy_state(state)
    async def delete(self, *, tenant_id: int, session_id: str) -> None:
        """Delete state for one tenant/session pair from the test store.

        Args:
            tenant_id (int):
                Tenant that owns the session being deleted.
            session_id (str):
                Session identifier within the tenant.

        Returns:
            None:
                Missing tenant/session pairs are ignored.
        """
        self._states.pop((tenant_id, session_id), None)

class RedisSessionMemory:
    """Redis-backed tenant-scoped session memory for production requests."""

    def __init__(self, redis: Redis, *, ttl_seconds: int = 60 * 60 * 4) -> None:
        """Create a Redis session memory adapter.

        Args:
            redis (Redis):
                Async Redis client used for JSON state storage.
            ttl_seconds (int):
                Expiration window for short-lived session memory records.

        Returns:
            None:
                The adapter stores the Redis client and TTL configuration.
        """
        self.redis = redis
        self.ttl_seconds = ttl_seconds

    async def load(self, *, tenant_id: int, session_id: str) -> SessionState:
        """Load tenant-scoped session state from Redis.

        Args:
            tenant_id (int):
                Tenant that owns the requested session.
            session_id (str):
                Session identifier within the tenant.

        Returns:
            SessionState:
                Parsed session state, or an empty state if the key is missing or
                contains malformed data.
        """
        payload = await self.redis.get(_key(tenant_id=tenant_id, session_id=session_id))
        if payload is None:
            return SessionState()
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        try:
            decoded = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return SessionState()
        if not isinstance(decoded, dict):
            return SessionState()
        return _state_from_dict(decoded)

    async def save(self, *, tenant_id: int, session_id: str, state: SessionState) -> None:
        """Serialize tenant-scoped session state to Redis.

        Args:
            tenant_id (int):
                Tenant that owns the session being saved.
            session_id (str):
                Session identifier within the tenant.
            state (SessionState):
                State to serialize as JSON.

        Returns:
            None:
                Redis is updated with a TTL-bound JSON payload.
        """
        await self.redis.set(
            _key(tenant_id=tenant_id, session_id=session_id),
            json.dumps(_state_to_dict(state)),
            ex=self.ttl_seconds,
        )
    async def delete(self, *, tenant_id: int, session_id: str) -> None:
        """Delete tenant-scoped session state from Redis.

        Args:
            tenant_id (int):
                Tenant that owns the session being deleted.
            session_id (str):
                Session identifier within the tenant.

        Returns:
            None:
                Redis receives a delete for the tenant/session key.
        """
        await self.redis.delete(_key(tenant_id=tenant_id, session_id=session_id))

def get_redis_session_memory() -> RedisSessionMemory:
    """Build the default Redis session-memory adapter from settings.

    Args:
        None:
            Configuration is read from application settings.

    Returns:
        RedisSessionMemory:
            Redis-backed session memory using the configured Redis URL.
    """
    from redis.asyncio import Redis

    return RedisSessionMemory(Redis.from_url(settings.redis_url, decode_responses=True))


def _key(*, tenant_id: int, session_id: str) -> str:
    """Build the Redis key for one tenant/session pair.

    Args:
        tenant_id (int):
            Tenant that owns the session memory.
        session_id (str):
            Session identifier within the tenant.

    Returns:
        str:
            Redis key that cannot collide across tenants for the same session ID.
    """
    return f"cafe_assistant:tenant:{tenant_id}:session:{session_id}"


def append_turns(
    state: SessionState,
    user_message: str,
    assistant_message: str,
    *,
    max_turns: int = 10,
) -> SessionState:
    """Append a user/assistant pair while preserving restrictions and preferences.

    Args:
        state (SessionState):
            Existing session state before the new turn pair is added.
        user_message (str):
            User message to append as an untrusted recent turn.
        assistant_message (str):
            Assistant response to append as the paired recent turn.
        max_turns (int):
            Maximum number of recent turn records to retain.

    Returns:
        SessionState:
            New session state with the latest turns retained and other state copied.
    """
    turns = [
        *state.recent_turns,
        ConversationTurn(role="user", content=user_message),
        ConversationTurn(role="assistant", content=assistant_message),
    ][-max_turns:]
    return SessionState(
        restrictions=state.restrictions,
        preferences=dict(state.preferences),
        recent_turns=turns,
    )


def _state_to_dict(state: SessionState) -> dict[str, object]:
    """Convert session state into a JSON-serializable dictionary.

    Args:
        state (SessionState):
            Session state to serialize for Redis.

    Returns:
        dict[str, object]:
            JSON-compatible representation of restrictions, preferences, and turns.
    """
    return {
        "restrictions": {
            "avoid_allergens": sorted(
                allergen.value for allergen in state.restrictions.avoid_allergens
            ),
            "modes": sorted(mode.value for mode in state.restrictions.modes),
            "prefer_low_sugar": state.restrictions.prefer_low_sugar,
        },
        "preferences": dict(state.preferences),
        "recent_turns": [
            {"role": turn.role, "content": turn.content} for turn in state.recent_turns[-10:]
        ],
    }


def _state_from_dict(payload: dict[str, object]) -> SessionState:
    """Parse a JSON dictionary into defensive session state.

    Args:
        payload (dict[str, object]):
            Decoded Redis payload. Unknown or malformed fields are ignored.

    Returns:
        SessionState:
            Parsed session state with invalid enum values dropped rather than raised.
    """
    restrictions_payload = payload.get("restrictions", {})
    if not isinstance(restrictions_payload, dict):
        restrictions_payload = {}
    turns_payload = payload.get("recent_turns", [])
    if not isinstance(turns_payload, list):
        turns_payload = []
    preferences_payload = payload.get("preferences", {})
    if not isinstance(preferences_payload, dict):
        preferences_payload = {}
    return SessionState(
        restrictions=CustomerRestrictions(
            avoid_allergens=_parse_allergens(restrictions_payload.get("avoid_allergens", [])),
            modes=_parse_modes(restrictions_payload.get("modes", [])),
            prefer_low_sugar=bool(restrictions_payload.get("prefer_low_sugar", False)),
        ),
        preferences={str(key): value for key, value in preferences_payload.items()},
        recent_turns=[
            ConversationTurn(role=str(turn["role"]), content=str(turn["content"]))
            for turn in turns_payload
            if isinstance(turn, dict) and "role" in turn and "content" in turn
        ],
    )


def _parse_allergens(value: object) -> set[AllergenCode]:
    """Parse allergen enum values from stored JSON.

    Args:
        value (object):
            Stored allergen list from Redis.

    Returns:
        set[AllergenCode]:
            Valid allergen codes, with unknown values ignored.
    """
    if not isinstance(value, list):
        return set()
    parsed: set[AllergenCode] = set()
    for code in value:
        try:
            parsed.add(AllergenCode(str(code)))
        except ValueError:
            continue
    return parsed


def _parse_modes(value: object) -> set[DietaryMode]:
    """Parse dietary mode enum values from stored JSON.

    Args:
        value (object):
            Stored dietary mode list from Redis.

    Returns:
        set[DietaryMode]:
            Valid dietary modes, with unknown values ignored.
    """
    if not isinstance(value, list):
        return set()
    parsed: set[DietaryMode] = set()
    for code in value:
        try:
            parsed.add(DietaryMode(str(code)))
        except ValueError:
            continue
    return parsed


def _copy_state(state: SessionState) -> SessionState:
    """Create a defensive copy of mutable session state.

    Args:
        state (SessionState):
            State object to copy before returning or storing in tests.

    Returns:
        SessionState:
            New state object with copied sets, dictionaries, and turn list.
    """
    return SessionState(
        restrictions=CustomerRestrictions(
            avoid_allergens=set(state.restrictions.avoid_allergens),
            modes=set(state.restrictions.modes),
            prefer_low_sugar=state.restrictions.prefer_low_sugar,
        ),
        preferences=dict(state.preferences),
        recent_turns=list(state.recent_turns),
    )