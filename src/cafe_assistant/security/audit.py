"""Append-only audit-event writer for significant governance actions.

Application code should create audit rows through `append_audit_event` only. There
are no update/delete helpers in this module. Long-term retention deletion, when
required by policy, is handled by the privileged cleanup script and should be run
as an operational governance job rather than as part of request handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.models import AuditEvent
from cafe_assistant.security.redaction import redact_payload


@dataclass(frozen=True, slots=True)
class AuditContext:
    """Tenant, actor, and trace metadata attached to one audit event.

    Attributes:
        tenant_id (int):
            Tenant that owns the action being audited.
        actor (str):
            Actor label such as `anonymous`, `admin`, or `customer:<id>`.
        request_id (str):
            Request ID propagated from the API boundary.
        trace_id (str):
            Trace ID used to correlate audit and observability data.
    """

    tenant_id: int
    actor: str
    request_id: str
    trace_id: str


async def append_audit_event(
    session: AsyncSession,
    *,
    context: AuditContext,
    action: str,
    payload: dict[str, Any] | None = None,
) -> AuditEvent:
    """Append one redacted audit event inside the current transaction.

    Args:
        session (AsyncSession):
            Async database session that will own the inserted audit row.
        context (AuditContext):
            Tenant, actor, request, and trace metadata for the event.
        action (str):
            Stable action label, such as `recommendation_served` or `profile_deleted`.
        payload (dict[str, Any] | None):
            Optional structured payload. Values are redacted before insertion.

    Returns:
        AuditEvent:
            ORM row added to the session and flushed to receive an ID.
    """
    event = AuditEvent(
        tenant_id=context.tenant_id,
        actor=context.actor,
        action=action,
        request_id=context.request_id,
        trace_id=context.trace_id,
        payload_redacted=redact_payload(payload or {}),
    )
    session.add(event)
    await session.flush()
    return event
