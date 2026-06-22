"""Apply governance retention windows for customer, session, and audit data.

This script is intended to be run as an operational governance job, not from
request-handling code. Customer deletion uses database cascades for durable
profile, consent, event, and device-token rows. Audit rows are append-only from
application code; purging old audit rows requires the explicit
`--purge-audit-events` flag so operators do not accidentally erase audit history.
"""

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
    purge_audit_events: bool = False,
) -> dict[str, int]:
    """Delete data older than configured retention windows.

    Args:
        session (AsyncSession):
            Async database session used for counts and deletes.
        profile_retention_days (int):
            Maximum age for durable customer rows before cascade deletion.
        session_retention_days (int):
            Maximum age for standalone episodic-event rows.
        audit_retention_days (int):
            Maximum age for audit rows eligible for privileged purge.
        dry_run (bool):
            When True, only counts eligible rows and performs no deletion.
        purge_audit_events (bool):
            When True, deletes audit rows past retention. This is intentionally
            opt-in because audit rows are otherwise append-only application data.

    Returns:
        dict[str, int]:
            Counts of rows eligible for each cleanup family and audit rows purged.
    """
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
        "audit_events_purged": 0,
    }
    if dry_run:
        return counts

    await session.execute(delete(Customer).where(Customer.created_at < profile_cutoff))
    await session.execute(delete(EpisodicEvent).where(EpisodicEvent.created_at < event_cutoff))
    if purge_audit_events:
        await session.execute(delete(AuditEvent).where(AuditEvent.created_at < audit_cutoff))
        counts["audit_events_purged"] = counts["audit_events"]
    await session.commit()
    return counts


async def main() -> None:
    """Parse CLI arguments and run the retention cleanup job.

    Args:
        None:
            Arguments are read from the process command line.

    Returns:
        None:
            Cleanup counts are printed to stdout for operator review.
    """
    parser = argparse.ArgumentParser(description="Apply retention cleanup windows.")
    parser.add_argument("--profile-days", type=int, default=settings.profile_retention_days)
    parser.add_argument("--session-days", type=int, default=settings.session_retention_days)
    parser.add_argument("--audit-days", type=int, default=settings.audit_retention_days)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--purge-audit-events",
        action="store_true",
        help="Actually delete audit rows older than --audit-days.",
    )
    args = parser.parse_args()

    async with async_session_maker() as session:
        counts = await cleanup_retention(
            session,
            profile_retention_days=args.profile_days,
            session_retention_days=args.session_days,
            audit_retention_days=args.audit_days,
            dry_run=args.dry_run,
            purge_audit_events=args.purge_audit_events,
        )
    print(counts)


async def _count(session: AsyncSession, statement: object) -> int:
    """Return an integer count for a SQLAlchemy scalar count statement.

    Args:
        session (AsyncSession):
            Async database session used to execute the count.
        statement (object):
            SQLAlchemy scalar statement returning a count-like value.

    Returns:
        int:
            Count value, normalized to zero when the database returns None.
    """
    count = await session.scalar(statement)
    return int(count or 0)


if __name__ == "__main__":
    asyncio.run(main())
