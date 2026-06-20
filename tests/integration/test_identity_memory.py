"""Tests for identity memory.
Exercises expected behavior with deterministic fixtures and mocked providers where needed.
"""

from __future__ import annotations

import math
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cafe_assistant.agent.state_machine import AgentConfig, AgentState, ChatAgent, ChatAgentRequest
from cafe_assistant.db.base import Base
from cafe_assistant.db.models import Location, Tenant
from cafe_assistant.db.repositories.consent_repo import DIETARY_HEALTH_SCOPE, grant_consent
from cafe_assistant.db.repositories.profile_repo import (
    delete_customer_profile,
    get_or_create_customer_by_phone,
    load_stored_profile,
    update_dietary_facts,
)
from cafe_assistant.domain.dietary import AllergenCode, CustomerRestrictions
from cafe_assistant.gateway.model_gateway import ChatMessage, ChatModelCascade
from cafe_assistant.identity.device import issue_device_token, verify_device_token
from cafe_assistant.identity.otp import OtpService, hash_phone
from cafe_assistant.memory.session import InMemorySessionMemory, SessionState
from tests.fixtures.legacy_embeddings import backfill_menu_embeddings
from tests.fixtures.legacy_menu import TENANT_NAME, seed_database


class FakeEmbeddingProvider:
    """Container for fake embedding provider behavior and data."""
    dimensions = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed the requested value.

        Args:
            texts (list[str]):
                Input texts that should each receive one embedding vector.

        Returns:
            list[list[float]]:
                Value produced for the caller according to the function contract.
        """
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Embed one.

        Args:
            text (str):
                Input text to normalize, embed, tokenize, or classify.

        Returns:
            list[float]:
                Value produced for the caller according to the function contract.
        """
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        vector = [
            self._feature(tokens, {"coffee", "espresso", "cappuccino", "latte", "mocha"}),
            self._feature(tokens, {"tea", "chai", "matcha", "earl"}),
            self._feature(tokens, {"almond", "almnd", "nut"}),
            self._feature(tokens, {"cookie", "peanut", "butter"}),
            self._feature(tokens, {"sandwich", "panini", "toast"}),
            self._feature(tokens, {"chocolate", "mocha"}),
            self._feature(tokens, {"gluten", "bread", "sourdough", "dough"}),
            self._feature(tokens, {"vegan", "vegetarian", "gluten_free"}),
        ]
        magnitude = math.sqrt(sum(component * component for component in vector))
        if magnitude == 0:
            return self._pad(vector)
        return self._pad([component / magnitude for component in vector])

    def _feature(self, tokens: set[str], vocabulary: set[str]) -> float:
        """Handle feature.

        Args:
            tokens (set[str]):
                Tokens value required to perform this operation.
            vocabulary (set[str]):
                Vocabulary value required to perform this operation.

        Returns:
            float:
                Value produced for the caller according to the function contract.
        """
        return float(len(tokens & vocabulary))

    def _pad(self, vector: list[float]) -> list[float]:
        """Handle pad.

        Args:
            vector (list[float]):
                Vector being normalized, converted, or sent to the vector store.

        Returns:
            list[float]:
                Value produced for the caller according to the function contract.
        """
        return vector + [0.0] * (self.dimensions - len(vector))


class CapturingChatProvider:
    """Container for capturing chat provider behavior and data."""
    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """Handle stream chat.

        Args:
            messages (list[ChatMessage]):
                Ordered chat messages sent to the configured chat provider.
            timeout_seconds (float):
                Maximum time allowed for the streaming chat request.

        Returns:
            AsyncIterator[str]:
                Streamed values yielded to the caller as they become available.
        """
        del timeout_seconds
        item_names = [
            line.removeprefix("SAFE_ITEM:").split("|", 1)[0].strip()
            for message in messages
            for line in message.content.splitlines()
            if line.startswith("SAFE_ITEM:")
        ]
        response = (
            "I can suggest " + ", ".join(item_names) + "."
            if item_names
            else "I cannot suggest a menu item from the provided safe set."
        )
        for chunk in _chunks(response):
            yield chunk


class CapturingSmsSender:
    """Container for capturing sms sender behavior and data."""
    def __init__(self) -> None:
        """Initialize the object with the dependencies or values required later.

        Args:
            None.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
        self.sent: list[tuple[str, str]] = []

    async def send_otp(self, phone: str, code: str) -> None:
        """Handle send OTP.

        Args:
            phone (str):
                Phone value required to perform this operation.
            code (str):
                Code value required to perform this operation.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
        self.sent.append((phone, code))


@dataclass(slots=True)
class IdentityFixture:
    """Container for identity fixture behavior and data."""
    session: AsyncSession
    tenant_id: int
    other_tenant_id: int
    agent: ChatAgent
    memory: InMemorySessionMemory


@pytest.fixture
async def identity_fixture() -> AsyncIterator[IdentityFixture]:
    """Handle identity fixture.

    Args:
        None.

    Returns:
        AsyncIterator[IdentityFixture]:
            Streamed values yielded to the caller as they become available.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        provider = FakeEmbeddingProvider()
        memory = InMemorySessionMemory()
        await seed_database(session)
        tenant_id = await session.scalar(select(Tenant.id).where(Tenant.name == TENANT_NAME))
        assert tenant_id is not None

        other_tenant = Tenant(name="Other Cafe")
        other_tenant.locations.append(Location(name="Other Counter"))
        session.add(other_tenant)
        await session.commit()
        other_tenant_id = other_tenant.id

        await backfill_menu_embeddings(session, provider=provider, tenant_id=tenant_id)
        agent = ChatAgent(
            session,
            memory=memory,
            chat_models=ChatModelCascade(
                cheap=CapturingChatProvider(),
                strong=CapturingChatProvider(),
            ),
            embedding_provider=provider,
            config=AgentConfig(max_tool_calls=4, deadline_seconds=10.0, search_k=6),
        )
        yield IdentityFixture(
            session=session,
            tenant_id=tenant_id,
            other_tenant_id=other_tenant_id,
            agent=agent,
            memory=memory,
        )

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def test_anonymous_chat_flow_remains_functional(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify that anonymous chat flow remains functional.

    Args:
        identity_fixture (IdentityFixture):
            Identity fixture value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    result = await identity_fixture.agent.run(
        ChatAgentRequest(
            session_id="anon",
            tenant_id=identity_fixture.tenant_id,
            message="Recommend a latte.",
        )
    )

    assert AgentState.COMPLETE in result.state_history
    assert result.customer_id is None
    assert result.safe_items


async def test_device_recognition_loads_profile_restrictions(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify that device recognition loads profile restrictions.

    Args:
        identity_fixture (IdentityFixture):
            Identity fixture value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    token = await _create_consented_customer_with_peanut_profile(identity_fixture)

    result = await identity_fixture.agent.run(
        ChatAgentRequest(
            session_id="returning",
            tenant_id=identity_fixture.tenant_id,
            device_token=token,
            message="Can I get a peanut butter cookie?",
        )
    )

    assert result.customer_id is not None
    assert "Peanut Butter Cookie" not in {item.name for item in result.safe_items}
    assert AllergenCode.PEANUT in result.restrictions.avoid_allergens


async def test_health_facts_are_gated_until_otp_consent(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify that health facts are gated until OTP consent.

    Args:
        identity_fixture (IdentityFixture):
            Identity fixture value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    session = identity_fixture.session
    tenant_id = identity_fixture.tenant_id

    customer = await get_or_create_customer_by_phone(
        session,
        tenant_id=tenant_id,
        phone_hash=hash_phone("+15555550100"),
    )
    token = await issue_device_token(session, tenant_id=tenant_id, customer_id=customer.id)

    await identity_fixture.agent.run(
        ChatAgentRequest(
            session_id="no-consent",
            tenant_id=tenant_id,
            device_token=token,
            message="I'm allergic to peanuts.",
        )
    )
    profile = await load_stored_profile(session, tenant_id=tenant_id, customer_id=customer.id)
    assert profile is not None
    assert profile.dietary_facts == {}

    otp_memory = SessionState(
        restrictions=CustomerRestrictions(
            avoid_allergens={AllergenCode.PEANUT},
            modes=set(),
        )
    )
    await identity_fixture.memory.save("otp-session", otp_memory)
    sms_sender = CapturingSmsSender()
    otp_service = OtpService(sender=sms_sender)

    start = await otp_service.start(tenant_id=tenant_id, phone="+15555550101")
    assert sms_sender.sent
    _sent_phone, code = sms_sender.sent[-1]
    confirmed = await otp_service.confirm(
        session,
        tenant_id=tenant_id,
        phone="+15555550101",
        challenge_id=start.challenge_id,
        code=code,
        session_state=await identity_fixture.memory.load("otp-session"),
    )

    consented_profile = await load_stored_profile(
        session,
        tenant_id=tenant_id,
        customer_id=confirmed.customer_id,
    )
    assert consented_profile is not None
    assert consented_profile.dietary_facts["avoid_allergens"] == ["PEANUT"]
    assert DIETARY_HEALTH_SCOPE in confirmed.granted_scopes


async def test_current_turn_override_beats_stored_profile_for_that_turn(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify that current turn override beats stored profile for that turn.

    Args:
        identity_fixture (IdentityFixture):
            Identity fixture value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    token = await _create_consented_customer_with_peanut_profile(identity_fixture)

    result = await identity_fixture.agent.run(
        ChatAgentRequest(
            session_id="override",
            tenant_id=identity_fixture.tenant_id,
            device_token=token,
            message="I'm not allergic to peanuts. Can I get a peanut butter cookie?",
        )
    )

    assert "Peanut Butter Cookie" in {item.name for item in result.safe_items}
    assert AllergenCode.PEANUT not in result.restrictions.avoid_allergens


async def test_profile_deletion_removes_device_access(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify that profile deletion removes device access.

    Args:
        identity_fixture (IdentityFixture):
            Identity fixture value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    token = await _create_consented_customer_with_peanut_profile(identity_fixture)
    identity = await verify_device_token(
        identity_fixture.session,
        tenant_id=identity_fixture.tenant_id,
        token=token,
    )
    assert identity is not None

    deleted = await delete_customer_profile(
        identity_fixture.session,
        tenant_id=identity_fixture.tenant_id,
        customer_id=identity.customer_id,
    )

    assert deleted is True
    removed_identity = await verify_device_token(
        identity_fixture.session,
        tenant_id=identity_fixture.tenant_id,
        token=token,
    )
    assert removed_identity is None


async def test_cross_tenant_access_is_blocked(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify that cross tenant access is blocked.

    Args:
        identity_fixture (IdentityFixture):
            Identity fixture value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    token = await _create_consented_customer_with_peanut_profile(identity_fixture)
    identity = await verify_device_token(
        identity_fixture.session,
        tenant_id=identity_fixture.tenant_id,
        token=token,
    )
    assert identity is not None

    cross_tenant_identity = await verify_device_token(
        identity_fixture.session,
        tenant_id=identity_fixture.other_tenant_id,
        token=token,
    )
    cross_tenant_profile = await load_stored_profile(
        identity_fixture.session,
        tenant_id=identity_fixture.other_tenant_id,
        customer_id=identity.customer_id,
    )

    assert cross_tenant_identity is None
    assert cross_tenant_profile is None


async def _create_consented_customer_with_peanut_profile(
    fixture: IdentityFixture,
) -> str:
    """Create consented customer with peanut profile.

    Args:
        fixture (IdentityFixture):
            Fixture value required to perform this operation.

    Returns:
        str:
            Value produced for the caller according to the function contract.
    """
    customer = await get_or_create_customer_by_phone(
        fixture.session,
        tenant_id=fixture.tenant_id,
        phone_hash=hash_phone("+15555550000"),
    )
    granted = await grant_consent(
        fixture.session,
        tenant_id=fixture.tenant_id,
        customer_id=customer.id,
        scope=DIETARY_HEALTH_SCOPE,
    )
    assert granted is True
    await update_dietary_facts(
        fixture.session,
        tenant_id=fixture.tenant_id,
        customer_id=customer.id,
        updates={"avoid_allergens": ["PEANUT"]},
    )
    return await issue_device_token(
        fixture.session,
        tenant_id=fixture.tenant_id,
        customer_id=customer.id,
    )


def _chunks(text: str, chunk_size: int = 12) -> list[str]:
    """Handle chunks.

    Args:
        text (str):
            Input text to normalize, embed, tokenize, or classify.
        chunk_size (int):
            Chunk size value required to perform this operation.

    Returns:
        list[str]:
            Value produced for the caller according to the function contract.
    """
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
