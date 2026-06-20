"""Tests for safe degradation.
Exercises expected behavior with deterministic fixtures and mocked providers where needed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cafe_assistant.agent.state_machine import AgentConfig, AgentState, ChatAgent, ChatAgentRequest
from cafe_assistant.db.base import Base
from cafe_assistant.db.models import Tenant
from cafe_assistant.domain.dietary import AllergenCode
from cafe_assistant.gateway.model_gateway import (
    ChatModelCascade,
    HashEmbeddingProvider,
    LocalChatProvider,
)
from cafe_assistant.memory.session import InMemorySessionMemory, SessionState
from tests.fixtures.legacy_embeddings import backfill_menu_embeddings
from tests.fixtures.legacy_menu import TENANT_NAME, seed_database


class FailingMemory:
    """Container for failing memory behavior and data."""
    async def load(self, session_id: str) -> SessionState:
        """Load the requested value.

        Args:
            session_id (str):
                Session id value required to perform this operation.

        Returns:
            SessionState:
                Value produced for the caller according to the function contract.
        """
        del session_id
        raise RuntimeError("memory unavailable")

    async def save(self, session_id: str, state: SessionState) -> None:
        """Handle save.

        Args:
            session_id (str):
                Session id value required to perform this operation.
            state (SessionState):
                State value required to perform this operation.

        Returns:
            None:
                No value is returned; the function completes through side effects or validation.
        """
        del session_id, state
        raise RuntimeError("memory unavailable")


@pytest.fixture
async def chaos_agent() -> AsyncIterator[tuple[ChatAgent, int]]:
    """Handle chaos agent.

    Args:
        None.

    Returns:
        AsyncIterator[tuple[ChatAgent, int]]:
            Streamed values yielded to the caller as they become available.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        await seed_database(session)
        tenant_id = await session.scalar(select(Tenant.id).where(Tenant.name == TENANT_NAME))
        assert tenant_id is not None
        await backfill_menu_embeddings(
            session,
            provider=HashEmbeddingProvider(384),
            tenant_id=tenant_id,
        )
        agent = ChatAgent(
            session,
            memory=InMemorySessionMemory(),
            chat_models=ChatModelCascade(
                cheap=LocalChatProvider(),
                strong=LocalChatProvider(),
            ),
            embedding_provider=HashEmbeddingProvider(384),
            config=AgentConfig(max_tool_calls=4, deadline_seconds=10.0, search_k=6),
        )
        yield agent, tenant_id

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def test_recommender_failure_falls_back_to_safe_menu_items(
    chaos_agent: tuple[ChatAgent, int],
) -> None:
    """Verify that recommender failure falls back to safe menu items.

    Args:
        chaos_agent (tuple[ChatAgent, int]):
            Chaos agent value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    agent, tenant_id = chaos_agent

    async def fail_search(input_model: BaseModel) -> BaseModel:
        """Handle fail search.

        Args:
            input_model (BaseModel):
                Input model value required to perform this operation.

        Returns:
            BaseModel:
                Value produced for the caller according to the function contract.
        """
        del input_model
        raise RuntimeError("retriever down")

    agent.tools._tools["search_menu"] = fail_search
    result = await agent.run(
        ChatAgentRequest(
            session_id="chaos-retriever",
            tenant_id=tenant_id,
            message="Recommend a coffee.",
        )
    )

    assert AgentState.FAILED not in result.state_history
    assert result.safe_items
    assert "I can suggest" in result.response


async def test_allergen_data_unavailable_uses_staff_check_fallback(
    chaos_agent: tuple[ChatAgent, int],
) -> None:
    """Verify that allergen data unavailable uses staff check fallback.

    Args:
        chaos_agent (tuple[ChatAgent, int]):
            Chaos agent value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    agent, tenant_id = chaos_agent

    result = await agent.run(
        ChatAgentRequest(
            session_id="chaos-allergen",
            tenant_id=tenant_id,
            message="I'm allergic to peanuts. Can I have the turkey pesto panini?",
        )
    )

    assert result.safe_items == []
    assert "can't confirm a safe option" in result.response
    assert "check with cafe staff" in result.response
    assert AllergenCode.PEANUT in result.restrictions.avoid_allergens


async def test_memory_unavailable_degrades_to_anonymous_session(
    chaos_agent: tuple[ChatAgent, int],
) -> None:
    """Verify that memory unavailable degrades to anonymous session.

    Args:
        chaos_agent (tuple[ChatAgent, int]):
            Chaos agent value required to perform this operation.

    Returns:
        None:
            No value is returned; failed expectations raise pytest assertion errors.
    """
    agent, tenant_id = chaos_agent
    agent.memory = FailingMemory()

    result = await agent.run(
        ChatAgentRequest(
            session_id="chaos-memory",
            tenant_id=tenant_id,
            message="Recommend a latte.",
        )
    )

    assert AgentState.FAILED not in result.state_history
    assert result.safe_items
    assert result.customer_id is None
