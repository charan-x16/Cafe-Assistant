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
    role: str
    content: str


@dataclass(slots=True)
class SessionState:
    restrictions: CustomerRestrictions = field(
        default_factory=lambda: CustomerRestrictions(
            avoid_allergens=set(),
            modes=set(),
            prefer_low_sugar=False,
        )
    )
    recent_turns: list[ConversationTurn] = field(default_factory=list)


class SessionMemory(Protocol):
    async def load(self, session_id: str) -> SessionState:
        """Load session-scoped memory."""

    async def save(self, session_id: str, state: SessionState) -> None:
        """Persist session-scoped memory."""


class InMemorySessionMemory:
    def __init__(self) -> None:
        self._states: dict[str, SessionState] = {}

    async def load(self, session_id: str) -> SessionState:
        return _copy_state(self._states.get(session_id, SessionState()))

    async def save(self, session_id: str, state: SessionState) -> None:
        self._states[session_id] = _copy_state(state)


class RedisSessionMemory:
    def __init__(self, redis: Redis, *, ttl_seconds: int = 60 * 60 * 4) -> None:
        self.redis = redis
        self.ttl_seconds = ttl_seconds

    async def load(self, session_id: str) -> SessionState:
        payload = await self.redis.get(_key(session_id))
        if payload is None:
            return SessionState()
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        return _state_from_dict(json.loads(payload))

    async def save(self, session_id: str, state: SessionState) -> None:
        await self.redis.set(
            _key(session_id),
            json.dumps(_state_to_dict(state)),
            ex=self.ttl_seconds,
        )


def get_redis_session_memory() -> RedisSessionMemory:
    from redis.asyncio import Redis

    return RedisSessionMemory(Redis.from_url(settings.redis_url, decode_responses=True))


def _key(session_id: str) -> str:
    return f"cafe_assistant:session:{session_id}"


def append_turns(
    state: SessionState,
    user_message: str,
    assistant_message: str,
    *,
    max_turns: int = 10,
) -> SessionState:
    turns = [
        *state.recent_turns,
        ConversationTurn(role="user", content=user_message),
        ConversationTurn(role="assistant", content=assistant_message),
    ][-max_turns:]
    return SessionState(restrictions=state.restrictions, recent_turns=turns)


def _state_to_dict(state: SessionState) -> dict[str, object]:
    return {
        "restrictions": {
            "avoid_allergens": sorted(
                allergen.value for allergen in state.restrictions.avoid_allergens
            ),
            "modes": sorted(mode.value for mode in state.restrictions.modes),
            "prefer_low_sugar": state.restrictions.prefer_low_sugar,
        },
        "recent_turns": [
            {"role": turn.role, "content": turn.content} for turn in state.recent_turns[-10:]
        ],
    }


def _state_from_dict(payload: dict[str, object]) -> SessionState:
    restrictions_payload = payload.get("restrictions", {})
    if not isinstance(restrictions_payload, dict):
        restrictions_payload = {}
    turns_payload = payload.get("recent_turns", [])
    if not isinstance(turns_payload, list):
        turns_payload = []
    return SessionState(
        restrictions=CustomerRestrictions(
            avoid_allergens={
                AllergenCode(code)
                for code in restrictions_payload.get("avoid_allergens", [])
            },
            modes={DietaryMode(code) for code in restrictions_payload.get("modes", [])},
            prefer_low_sugar=bool(restrictions_payload.get("prefer_low_sugar", False)),
        ),
        recent_turns=[
            ConversationTurn(role=str(turn["role"]), content=str(turn["content"]))
            for turn in turns_payload
            if isinstance(turn, dict) and "role" in turn and "content" in turn
        ],
    )


def _copy_state(state: SessionState) -> SessionState:
    return SessionState(
        restrictions=CustomerRestrictions(
            avoid_allergens=set(state.restrictions.avoid_allergens),
            modes=set(state.restrictions.modes),
            prefer_low_sugar=state.restrictions.prefer_low_sugar,
        ),
        recent_turns=list(state.recent_turns),
    )
