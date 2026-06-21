"""Integration tests for identity, durable profile, and session memory.

The suite uses deterministic fake providers and an in-memory database to verify
anonymous sessions, tenant-scoped session memory, OTP consent upgrades, durable
profile reads/writes, current-turn overrides, and deletion behavior.
"""

from __future__ import annotations

import math
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cafe_assistant.agent.state_machine import AgentConfig, AgentState, ChatAgent, ChatAgentRequest
from cafe_assistant.db.base import Base
from cafe_assistant.db.models import EpisodicEvent, Location, Tenant
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
    """Deterministic embedding provider used by identity-memory chat tests.

    The provider maps recognizable menu vocabulary to stable vectors so tests
    can exercise retrieval without external model calls.
    """
    dimensions = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each text with the deterministic test vectorizer.

        Args:
            texts (list[str]):
                Query or menu texts that should each receive one embedding vector.

        Returns:
            list[list[float]]:
                One fixed-width vector per input text, in the same order.
        """
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Embed one text by counting known menu vocabulary groups.

        Args:
            text (str):
                Query or menu text to tokenize and convert into feature counts.

        Returns:
            list[float]:
                Normalized and padded embedding vector for the supplied text.
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
        """Count how many recognized vocabulary terms appear in the text.

        Args:
            tokens (set[str]):
                Lowercase tokens extracted from one input text.
            vocabulary (set[str]):
                Menu-topic terms represented by one vector dimension.

        Returns:
            float:
                Count of matching tokens as a numeric feature value.
        """
        return float(len(tokens & vocabulary))

    def _pad(self, vector: list[float]) -> list[float]:
        """Pad a short feature vector to the configured embedding dimension.

        Args:
            vector (list[float]):
                Normalized feature values produced for one text.

        Returns:
            list[float]:
                Vector extended with zeros to match the fake provider dimension.
        """
        return vector + [0.0] * (self.dimensions - len(vector))


class CapturingChatProvider:
    """Chat provider double that streams responses from supplied safe items."""
    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """Stream deterministic text naming only `SAFE_ITEM` prompt entries.

        Args:
            messages (list[ChatMessage]):
                Ordered chat messages built by the composer.
            timeout_seconds (float):
                Maximum time allowed for streaming; accepted but unused by the fake.

        Returns:
            AsyncIterator[str]:
                Response chunks grounded in safe item names from the prompt.
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
    """SMS sender double that captures OTP codes for confirmation tests."""
    def __init__(self) -> None:
        """Initialize an empty list of sent OTP messages.

        Args:
            None:
                The fake sender has no external dependencies.

        Returns:
            None:
                The `sent` list is ready for assertions.
        """
        self.sent: list[tuple[str, str]] = []

    async def send_otp(self, phone: str, code: str) -> None:
        """Capture an OTP send request for test confirmation.

        Args:
            phone (str):
                Normalized destination phone number.
            code (str):
                One-time code generated by the OTP service.

        Returns:
            None:
                The phone/code pair is appended to the capture list.
        """
        self.sent.append((phone, code))


@dataclass(slots=True)
class IdentityFixture:
    """Seeded identity-memory fixture values shared by integration tests."""
    session: AsyncSession
    tenant_id: int
    other_tenant_id: int
    agent: ChatAgent
    memory: InMemorySessionMemory


@pytest.fixture
async def identity_fixture() -> AsyncIterator[IdentityFixture]:
    """Create a seeded identity-memory fixture with two tenants.

    Args:
        None:
            Pytest manages fixture setup and teardown.

    Returns:
        AsyncIterator[IdentityFixture]:
            Database session, tenant IDs, chat agent, and tenant-scoped session memory.
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
            Seeded fixture containing tenants, database session, memory, and chat agent.

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
            Seeded fixture containing tenants, database session, memory, and chat agent.

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
            Seeded fixture containing tenants, database session, memory, and chat agent.

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
    await identity_fixture.memory.save(
        tenant_id=tenant_id,
        session_id="otp-session",
        state=otp_memory,
    )
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
        session_state=await identity_fixture.memory.load(
            tenant_id=tenant_id,
            session_id="otp-session",
        ),
    )

    consented_profile = await load_stored_profile(
        session,
        tenant_id=tenant_id,
        customer_id=confirmed.customer_id,
    )
    assert consented_profile is not None
    assert consented_profile.dietary_facts["avoid_allergens"] == ["PEANUT"]
    assert DIETARY_HEALTH_SCOPE in confirmed.granted_scopes


async def test_session_memory_is_tenant_scoped_for_same_session_id(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify same session IDs do not share memory across tenants.

    Args:
        identity_fixture (IdentityFixture):
            Seeded identity fixture with one primary tenant and one second tenant.

    Returns:
        None:
            Failed expectations raise pytest assertion errors.
    """
    shared_session_id = "shared-browser-session"

    await identity_fixture.agent.run(
        ChatAgentRequest(
            session_id=shared_session_id,
            tenant_id=identity_fixture.tenant_id,
            message="I'm allergic to peanuts.",
        )
    )

    primary_state = await identity_fixture.memory.load(
        tenant_id=identity_fixture.tenant_id,
        session_id=shared_session_id,
    )
    other_state = await identity_fixture.memory.load(
        tenant_id=identity_fixture.other_tenant_id,
        session_id=shared_session_id,
    )
    other_result = await identity_fixture.agent.run(
        ChatAgentRequest(
            session_id=shared_session_id,
            tenant_id=identity_fixture.other_tenant_id,
            message="Recommend a latte.",
        )
    )

    assert AllergenCode.PEANUT in primary_state.restrictions.avoid_allergens
    assert AllergenCode.PEANUT not in other_state.restrictions.avoid_allergens
    assert AllergenCode.PEANUT not in other_result.restrictions.avoid_allergens


async def test_otp_upgrade_uses_tenant_scoped_session_memory(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify OTP upgrade cannot import another tenant's pending session facts.

    Args:
        identity_fixture (IdentityFixture):
            Seeded identity fixture with isolated tenant-scoped session memory.

    Returns:
        None:
            Failed expectations raise pytest assertion errors.
    """
    shared_session_id = "shared-otp-session"
    await identity_fixture.memory.save(
        tenant_id=identity_fixture.tenant_id,
        session_id=shared_session_id,
        state=SessionState(
            restrictions=CustomerRestrictions(
                avoid_allergens={AllergenCode.PEANUT},
                modes=set(),
            )
        ),
    )

    other_tenant_state = await identity_fixture.memory.load(
        tenant_id=identity_fixture.other_tenant_id,
        session_id=shared_session_id,
    )
    sms_sender = CapturingSmsSender()
    otp_service = OtpService(sender=sms_sender)
    start = await otp_service.start(
        tenant_id=identity_fixture.other_tenant_id,
        phone="+15555550177",
    )
    _sent_phone, code = sms_sender.sent[-1]
    confirmed = await otp_service.confirm(
        identity_fixture.session,
        tenant_id=identity_fixture.other_tenant_id,
        phone="+15555550177",
        challenge_id=start.challenge_id,
        code=code,
        session_state=other_tenant_state,
    )

    profile = await load_stored_profile(
        identity_fixture.session,
        tenant_id=identity_fixture.other_tenant_id,
        customer_id=confirmed.customer_id,
    )

    assert profile is not None
    assert profile.dietary_facts == {}


async def test_write_gate_does_not_persist_merged_profile_facts_on_unrelated_turn(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify durable writes use explicit current-turn facts, not merged memory.

    Args:
        identity_fixture (IdentityFixture):
            Seeded identity fixture with a consented peanut-allergic profile.

    Returns:
        None:
            Failed expectations raise pytest assertion errors.
    """
    token = await _create_consented_customer_with_peanut_profile(identity_fixture)
    identity = await verify_device_token(
        identity_fixture.session,
        tenant_id=identity_fixture.tenant_id,
        token=token,
    )
    assert identity is not None
    before_count = await identity_fixture.session.scalar(
        select(func.count())
        .select_from(EpisodicEvent)
        .where(
            EpisodicEvent.customer_id == identity.customer_id,
            EpisodicEvent.type == "dietary_facts_saved",
        )
    )

    result = await identity_fixture.agent.run(
        ChatAgentRequest(
            session_id="profile-fact-not-rewritten",
            tenant_id=identity_fixture.tenant_id,
            device_token=token,
            message="Recommend a latte.",
        )
    )

    after_count = await identity_fixture.session.scalar(
        select(func.count())
        .select_from(EpisodicEvent)
        .where(
            EpisodicEvent.customer_id == identity.customer_id,
            EpisodicEvent.type == "dietary_facts_saved",
        )
    )

    assert AgentState.COMPLETE in result.state_history
    assert AllergenCode.PEANUT in result.restrictions.avoid_allergens
    assert after_count == before_count


async def test_anonymous_session_preference_is_reused_without_profile(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify anonymous session preferences are remembered during the visit.

    Args:
        identity_fixture (IdentityFixture):
            Seeded identity fixture with in-memory session storage.

    Returns:
        None:
            Failed expectations raise pytest assertion errors.
    """
    session_id = "anonymous-oat-session"

    await identity_fixture.agent.run(
        ChatAgentRequest(
            session_id=session_id,
            tenant_id=identity_fixture.tenant_id,
            message="I prefer oat milk.",
        )
    )
    stored_state = await identity_fixture.memory.load(
        tenant_id=identity_fixture.tenant_id,
        session_id=session_id,
    )
    result = await identity_fixture.agent.run(
        ChatAgentRequest(
            session_id=session_id,
            tenant_id=identity_fixture.tenant_id,
            message="Recommend a latte.",
        )
    )
    model_context = "\n".join(message.content for message in result.model_messages)

    assert result.customer_id is None
    assert stored_state.preferences["milk_preference"] == "oat milk"
    assert "oat milk" in model_context

async def test_current_turn_override_beats_stored_profile_for_that_turn(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify that current turn override beats stored profile for that turn.

    Args:
        identity_fixture (IdentityFixture):
            Seeded fixture containing tenants, database session, memory, and chat agent.

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
            Seeded fixture containing tenants, database session, memory, and chat agent.

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
            Seeded fixture containing tenants, database session, memory, and chat agent.

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
    """Create a recognized customer with consented peanut allergy facts.

    Args:
        fixture (IdentityFixture):
            Seeded identity-memory fixture used to create the profile.

        Returns:
            str:
                Device token linked to the consented profile.
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
    """Split fake model output into deterministic streaming chunks.

    Args:
        text (str):
            Full fake response text to stream back to the caller.
        chunk_size (int):
            Maximum number of characters per emitted chunk.

        Returns:
            list[str]:
                Ordered chunks that reconstruct `text` when joined.
    """
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
