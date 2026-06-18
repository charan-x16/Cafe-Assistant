from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cafe_assistant.agent import state_machine
from cafe_assistant.api.deps import get_rate_limiter
from cafe_assistant.db.base import Base
from cafe_assistant.db.session import get_session
from cafe_assistant.main import create_app
from cafe_assistant.memory.session import InMemorySessionMemory
from cafe_assistant.security.rate_limit import InMemoryRateLimiter
from scripts.embed_menu import backfill_menu_embeddings
from scripts.seed_menu import seed_database


@pytest.mark.load
async def test_chat_first_token_latency_budget_under_burst(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        state_machine,
        "get_redis_session_memory",
        lambda: InMemorySessionMemory(),
    )
    db_path = tmp_path / "load.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    try:
        async with session_factory() as session:
            await seed_database(session)
            await backfill_menu_embeddings(session)
            app = create_app()
            app.dependency_overrides[get_rate_limiter] = lambda: InMemoryRateLimiter(
                session_limit=1_000,
                ip_limit=1_000,
            )

            async def override_get_session() -> AsyncIterator[AsyncSession]:
                async with session_factory() as request_session:
                    try:
                        yield request_session
                        await request_session.commit()
                    except Exception:
                        await request_session.rollback()
                        raise

            app.dependency_overrides[get_session] = override_get_session
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
                timeout=10.0,
                ) as client:
                    latencies = await asyncio.gather(
                        *[
                            _first_token_latency_ms(client, index)
                            for index in range(12)
                        ]
                    )

            p95 = _percentile(latencies, 0.95)
            p99 = _percentile(latencies, 0.99)
            assert p95 < 2_000
            assert p99 < 2_500
    finally:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await engine.dispose()


async def _first_token_latency_ms(client: AsyncClient, index: int) -> float:
    started_at = time.perf_counter()
    async with client.stream(
        "POST",
        "/chat",
        json={
            "session_id": f"load-{index}",
            "tenant_id": 1,
            "message": "Recommend a latte." if index % 2 else "Do you have an almnd latte?",
        },
    ) as response:
        async for chunk in response.aiter_text():
            if chunk:
                break
    return (time.perf_counter() - started_at) * 1000.0


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]
