from __future__ import annotations

import math
import re
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cafe_assistant.agent.state_machine import AgentConfig, AgentState, ChatAgent, ChatAgentRequest
from cafe_assistant.db.base import Base
from cafe_assistant.db.models import Tenant
from cafe_assistant.gateway.model_gateway import ChatMessage, ChatModelCascade
from cafe_assistant.memory.session import InMemorySessionMemory
from scripts.embed_menu import backfill_menu_embeddings
from scripts.seed_menu import TENANT_NAME, seed_database


class FakeEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
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
            return vector
        return [component / magnitude for component in vector]

    def _feature(self, tokens: set[str], vocabulary: set[str]) -> float:
        return float(len(tokens & vocabulary))


class CapturingChatProvider:
    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []
        self.safe_item_lines_by_call: list[list[str]] = []

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        del timeout_seconds
        self.calls.append(messages)
        safe_lines = [
            line.removeprefix("SAFE_ITEM:").strip()
            for message in messages
            for line in message.content.splitlines()
            if line.startswith("SAFE_ITEM:")
        ]
        self.safe_item_lines_by_call.append(safe_lines)
        item_names = [line.split("|", 1)[0].strip() for line in safe_lines]
        response = (
            "I can suggest " + ", ".join(item_names) + "."
            if item_names
            else "I cannot suggest a menu item from the provided safe set."
        )
        for chunk in _chunks(response):
            yield chunk


@pytest.fixture
async def chat_fixture() -> AsyncIterator[tuple[ChatAgent, int, CapturingChatProvider]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        embedding_provider = FakeEmbeddingProvider()
        strong_model = CapturingChatProvider()
        cheap_model = CapturingChatProvider()
        memory = InMemorySessionMemory()
        await seed_database(session)
        tenant_id = await session.scalar(select(Tenant.id).where(Tenant.name == TENANT_NAME))
        assert tenant_id is not None
        await backfill_menu_embeddings(session, provider=embedding_provider, tenant_id=tenant_id)
        agent = ChatAgent(
            session,
            memory=memory,
            chat_models=ChatModelCascade(cheap=cheap_model, strong=strong_model),
            embedding_provider=embedding_provider,
            config=AgentConfig(max_tool_calls=4, deadline_seconds=10.0, search_k=6),
        )
        yield agent, tenant_id, strong_model

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def test_peanut_allergy_persists_and_blocks_peanut_recommendations(
    chat_fixture: tuple[ChatAgent, int, CapturingChatProvider],
) -> None:
    agent, tenant_id, strong_model = chat_fixture

    await agent.run(
        ChatAgentRequest(
            session_id="peanut-session",
            tenant_id=tenant_id,
            message="I'm allergic to peanuts.",
        )
    )
    result = await agent.run(
        ChatAgentRequest(
            session_id="peanut-session",
            tenant_id=tenant_id,
            message="Can I get a peanut butter cookie?",
        )
    )

    assert "Peanut Butter Cookie" not in result.response
    assert all(item.name != "Peanut Butter Cookie" for item in result.safe_items)
    assert _latest_safe_item_names(strong_model) <= {item.name for item in result.safe_items}
    assert "Peanut Butter Cookie" not in _latest_safe_item_names(strong_model)


async def test_fuzzy_request_returns_grounded_menu_item(
    chat_fixture: tuple[ChatAgent, int, CapturingChatProvider],
) -> None:
    agent, tenant_id, _strong_model = chat_fixture

    result = await agent.run(
        ChatAgentRequest(
            session_id="fuzzy-session",
            tenant_id=tenant_id,
            message="Do you have an almnd latte?",
        )
    )

    assert "Matcha Almond Latte" in {item.name for item in result.safe_items}
    assert "Matcha Almond Latte" in result.response
    assert AgentState.COMPLETE in result.state_history


async def test_empty_safe_set_uses_staff_check_fallback(
    chat_fixture: tuple[ChatAgent, int, CapturingChatProvider],
) -> None:
    agent, tenant_id, strong_model = chat_fixture

    await agent.run(
        ChatAgentRequest(
            session_id="empty-session",
            tenant_id=tenant_id,
            message="I'm allergic to peanuts, tree nuts, dairy, gluten, soy, and eggs.",
        )
    )
    result = await agent.run(
        ChatAgentRequest(
            session_id="empty-session",
            tenant_id=tenant_id,
            message="Can I have the turkey pesto panini?",
        )
    )

    assert result.safe_items == []
    assert "can't confirm a safe option" in result.response
    assert "check with cafe staff" in result.response
    assert not strong_model.safe_item_lines_by_call or all(
        "Turkey Pesto Panini" not in line
        for safe_lines in strong_model.safe_item_lines_by_call
        for line in safe_lines
    )


async def test_medical_question_is_escalated_with_disclaimer(
    chat_fixture: tuple[ChatAgent, int, CapturingChatProvider],
) -> None:
    agent, tenant_id, strong_model = chat_fixture

    result = await agent.run(
        ChatAgentRequest(
            session_id="medical-session",
            tenant_id=tenant_id,
            message="How much insulin should I take for a mocha?",
        )
    )

    assert AgentState.ESCALATED in result.state_history
    assert "not medical advice" in result.response
    assert "insulin" in result.response
    assert strong_model.calls == []


async def test_model_context_contains_only_safe_items(
    chat_fixture: tuple[ChatAgent, int, CapturingChatProvider],
) -> None:
    agent, tenant_id, strong_model = chat_fixture

    result = await agent.run(
        ChatAgentRequest(
            session_id="context-session",
            tenant_id=tenant_id,
            message="I'm vegan and allergic to dairy. Recommend a latte.",
        )
    )

    safe_item_names = {item.name for item in result.safe_items}
    context_item_names = _latest_safe_item_names(strong_model)
    assert context_item_names
    assert context_item_names <= safe_item_names
    assert "Cappuccino" not in context_item_names
    assert "Mocha" not in context_item_names


def _latest_safe_item_names(provider: CapturingChatProvider) -> set[str]:
    if not provider.safe_item_lines_by_call:
        return set()
    return {
        line.split("|", 1)[0].strip()
        for line in provider.safe_item_lines_by_call[-1]
    }


def _chunks(text: str, chunk_size: int = 12) -> list[str]:
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
