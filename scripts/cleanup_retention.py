from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.config import settings
from cafe_assistant.db.models import AuditEvent, Customer, EpisodicEvent
from cafe_assistant.db.session import async_session_maker


async def cleanup_retention(
    session: AsyncSession,
    *,
    profile_retention_days: int = settings.profile_retention_days,
    session_retention_days: int = settings.session_retention_days,
    audit_retention_days: int = settings.audit_retention_days,
    dry_run: bool = False,
) -> dict[str, int]:
    now = datetime.now(UTC)
    profile_cutoff = now - timedelta(days=profile_retention_days)
    event_cutoff = now - timedelta(days=session_retention_days)
    audit_cutoff = now - timedelta(days=audit_retention_days)

    counts = {
        "customers": await _count(
            session,
            select(func.count()).select_from(Customer).where(Customer.created_at < profile_cutoff),
        ),
        "episodic_events": await _count(
            session,
            select(func.count())
            .select_from(EpisodicEvent)
            .where(EpisodicEvent.created_at < event_cutoff),
        ),
        "audit_events": await _count(
            session,
            select(func.count())
            .select_from(AuditEvent)
            .where(AuditEvent.created_at < audit_cutoff),
        ),
    }
    if dry_run:
        return counts

    await session.execute(delete(Customer).where(Customer.created_at < profile_cutoff))
    await session.execute(delete(EpisodicEvent).where(EpisodicEvent.created_at < event_cutoff))
    await session.execute(delete(AuditEvent).where(AuditEvent.created_at < audit_cutoff))
    await session.commit()
    return counts


async def main() -> None:
    parser = argparse.ArgumentParser(description="Apply retention cleanup windows.")
    parser.add_argument("--profile-days", type=int, default=settings.profile_retention_days)
    parser.add_argument("--session-days", type=int, default=settings.session_retention_days)
    parser.add_argument("--audit-days", type=int, default=settings.audit_retention_days)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    async with async_session_maker() as session:
        counts = await cleanup_retention(
            session,
            profile_retention_days=args.profile_days,
            session_retention_days=args.session_days,
            audit_retention_days=args.audit_days,
            dry_run=args.dry_run,
        )
    print(counts)


async def _count(session: AsyncSession, statement: object) -> int:
    count = await session.scalar(statement)
    return int(count or 0)


if __name__ == "__main__":
    asyncio.run(main())
