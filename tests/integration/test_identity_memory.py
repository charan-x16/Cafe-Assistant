"""Integration tests for identity, durable profile, and session memory.

The suite uses deterministic fake providers and an in-memory database to verify
anonymous sessions, tenant-scoped session memory, account consent grants, durable
profile reads/writes, current-turn overrides, and deletion behavior.
"""

from __future__ import annotations

import math
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cafe_assistant.agent.state_machine import AgentConfig, AgentState, ChatAgent, ChatAgentRequest
from cafe_assistant.db.base import Base
from cafe_assistant.db.models import Consent, CustomerDeviceToken, EpisodicEvent, Location, Tenant
from cafe_assistant.db.repositories.consent_repo import DIETARY_HEALTH_SCOPE, grant_consent
from cafe_assistant.db.repositories.profile_repo import (
    delete_customer_profile,
    load_stored_profile,
    update_dietary_facts,
)
from cafe_assistant.domain.dietary import AllergenCode, CustomerRestrictions
from cafe_assistant.gateway.model_gateway import ChatMessage, ChatModelCascade
from cafe_assistant.identity.account import create_customer_account
from cafe_assistant.identity.device import (
    issue_device_token,
    revoke_device_token,
    verify_device_token,
)
from cafe_assistant.identity.dietary_facts import restrictions_to_dietary_facts
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


async def test_health_facts_are_gated_until_account_health_consent(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify that health facts persist only after explicit account consent.

    Args:
        identity_fixture (IdentityFixture):
            Seeded fixture containing tenants, database session, memory, and chat agent.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    session = identity_fixture.session
    tenant_id = identity_fixture.tenant_id
    credentials = await create_customer_account(
        session,
        tenant_id=tenant_id,
        username="no-consent-user",
        password="password123",
    )
    token = await issue_device_token(
        session,
        tenant_id=tenant_id,
        customer_id=credentials.customer_id,
    )

    await identity_fixture.agent.run(
        ChatAgentRequest(
            session_id="no-consent",
            tenant_id=tenant_id,
            device_token=token,
            message="I'm allergic to peanuts.",
        )
    )
    profile = await load_stored_profile(
        session,
        tenant_id=tenant_id,
        customer_id=credentials.customer_id,
    )
    assert profile is not None
    assert profile.dietary_facts == {}

    await identity_fixture.memory.save(
        tenant_id=tenant_id,
        session_id="account-consent-session",
        state=SessionState(
            restrictions=CustomerRestrictions(
                avoid_allergens={AllergenCode.PEANUT},
                modes=set(),
            )
        ),
    )
    granted = await grant_consent(
        session,
        tenant_id=tenant_id,
        customer_id=credentials.customer_id,
        scope=DIETARY_HEALTH_SCOPE,
    )
    consent_state = await identity_fixture.memory.load(
        tenant_id=tenant_id,
        session_id="account-consent-session",
    )
    persisted = await update_dietary_facts(
        session,
        tenant_id=tenant_id,
        customer_id=credentials.customer_id,
        updates=restrictions_to_dietary_facts(consent_state.restrictions),
    )

    consented_profile = await load_stored_profile(
        session,
        tenant_id=tenant_id,
        customer_id=credentials.customer_id,
    )
    assert granted is True
    assert persisted is True
    assert consented_profile is not None
    assert consented_profile.dietary_facts["avoid_allergens"] == ["PEANUT"]

async def test_device_token_expiry_and_revocation(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify expired and revoked device tokens no longer recognize customers.

    Args:
        identity_fixture (IdentityFixture):
            Seeded fixture containing a database session and tenant ID.

    Returns:
        None:
            Failed expectations raise pytest assertion errors.
    """
    session = identity_fixture.session
    credentials = await create_customer_account(
        session,
        tenant_id=identity_fixture.tenant_id,
        username="token-expiry-user",
        password="password123",
    )
    customer_id = credentials.customer_id
    expired_token = await issue_device_token(
        session,
        tenant_id=identity_fixture.tenant_id,
        customer_id=customer_id,
    )
    token_row = await session.scalar(
        select(CustomerDeviceToken).where(CustomerDeviceToken.customer_id == customer_id)
    )
    assert token_row is not None
    token_row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await session.flush()

    expired_identity = await verify_device_token(
        session,
        tenant_id=identity_fixture.tenant_id,
        token=expired_token,
    )
    assert expired_identity is None

    active_token = await issue_device_token(
        session,
        tenant_id=identity_fixture.tenant_id,
        customer_id=customer_id,
    )
    active_identity = await verify_device_token(
        session,
        tenant_id=identity_fixture.tenant_id,
        token=active_token,
    )
    assert active_identity is not None

    revoked = await revoke_device_token(
        session,
        tenant_id=identity_fixture.tenant_id,
        token=active_token,
    )
    revoked_identity = await verify_device_token(
        session,
        tenant_id=identity_fixture.tenant_id,
        token=active_token,
    )
    assert revoked is True
    assert revoked_identity is None


async def test_duplicate_active_consent_is_blocked_by_database(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify the database blocks duplicate active consent rows.

    Args:
        identity_fixture (IdentityFixture):
            Seeded fixture containing a database session and tenant ID.

    Returns:
        None:
            Failed expectations raise pytest assertion errors.
    """
    session = identity_fixture.session
    credentials = await create_customer_account(
        session,
        tenant_id=identity_fixture.tenant_id,
        username="duplicate-consent-user",
        password="password123",
    )
    customer_id = credentials.customer_id
    granted = await grant_consent(
        session,
        tenant_id=identity_fixture.tenant_id,
        customer_id=customer_id,
        scope=DIETARY_HEALTH_SCOPE,
    )
    assert granted is True
    await session.commit()

    session.add(Consent(customer_id=customer_id, scope=DIETARY_HEALTH_SCOPE))
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()

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


async def test_health_consent_uses_tenant_scoped_session_memory(
    identity_fixture: IdentityFixture,
) -> None:
    """Verify account consent cannot import another tenant's session facts.

    Args:
        identity_fixture (IdentityFixture):
            Seeded identity fixture with isolated tenant-scoped session memory.

    Returns:
        None:
            Failed expectations raise pytest assertion errors.
    """
    shared_session_id = "shared-account-consent-session"
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
    credentials = await create_customer_account(
        identity_fixture.session,
        tenant_id=identity_fixture.other_tenant_id,
        username="other-tenant-consent-user",
        password="password123",
    )
    granted = await grant_consent(
        identity_fixture.session,
        tenant_id=identity_fixture.other_tenant_id,
        customer_id=credentials.customer_id,
        scope=DIETARY_HEALTH_SCOPE,
    )
    other_tenant_state = await identity_fixture.memory.load(
        tenant_id=identity_fixture.other_tenant_id,
        session_id=shared_session_id,
    )
    facts = restrictions_to_dietary_facts(other_tenant_state.restrictions)
    if facts:
        await update_dietary_facts(
            identity_fixture.session,
            tenant_id=identity_fixture.other_tenant_id,
            customer_id=credentials.customer_id,
            updates=facts,
        )

    profile = await load_stored_profile(
        identity_fixture.session,
        tenant_id=identity_fixture.other_tenant_id,
        customer_id=credentials.customer_id,
    )

    assert granted is True
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
    credentials = await create_customer_account(
        fixture.session,
        tenant_id=fixture.tenant_id,
        username="peanut-profile-user",
        password="password123",
    )
    customer_id = credentials.customer_id
    granted = await grant_consent(
        fixture.session,
        tenant_id=fixture.tenant_id,
        customer_id=customer_id,
        scope=DIETARY_HEALTH_SCOPE,
    )
    assert granted is True
    await update_dietary_facts(
        fixture.session,
        tenant_id=fixture.tenant_id,
        customer_id=customer_id,
        updates={"avoid_allergens": ["PEANUT"]},
    )
    return await issue_device_token(
        fixture.session,
        tenant_id=fixture.tenant_id,
        customer_id=customer_id,
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
