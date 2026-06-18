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
from cafe_assistant.gateway.model_gateway import ChatModelCascade, LocalChatProvider
from cafe_assistant.memory.session import InMemorySessionMemory, SessionState
from scripts.embed_menu import backfill_menu_embeddings
from scripts.seed_menu import TENANT_NAME, seed_database


class FailingMemory:
    async def load(self, session_id: str) -> SessionState:
        del session_id
        raise RuntimeError("memory unavailable")

    async def save(self, session_id: str, state: SessionState) -> None:
        del session_id, state
        raise RuntimeError("memory unavailable")


@pytest.fixture
async def chaos_agent() -> AsyncIterator[tuple[ChatAgent, int]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        await seed_database(session)
        tenant_id = await session.scalar(select(Tenant.id).where(Tenant.name == TENANT_NAME))
        assert tenant_id is not None
        await backfill_menu_embeddings(session, tenant_id=tenant_id)
        agent = ChatAgent(
            session,
            memory=InMemorySessionMemory(),
            chat_models=ChatModelCascade(
                cheap=LocalChatProvider(),
                strong=LocalChatProvider(),
            ),
            config=AgentConfig(max_tool_calls=4, deadline_seconds=10.0, search_k=6),
        )
        yield agent, tenant_id

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def test_recommender_failure_falls_back_to_safe_menu_items(
    chaos_agent: tuple[ChatAgent, int],
) -> None:
    agent, tenant_id = chaos_agent

    async def fail_search(input_model: BaseModel) -> BaseModel:
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
