from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.db.models import AuditEvent
from cafe_assistant.security.redaction import redact_payload


@dataclass(frozen=True, slots=True)
class AuditContext:
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
