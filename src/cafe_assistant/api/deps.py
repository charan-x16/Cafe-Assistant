from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from cafe_assistant.db.session import get_session
from cafe_assistant.identity.qr import InvalidQrPayloadError, parse_tenant_context
from cafe_assistant.security.rate_limit import (
    RateLimiter,
    RateLimitExceededError,
    get_redis_rate_limiter,
)

SessionDependency = Annotated[AsyncSession, Depends(get_session)]


@dataclass(frozen=True, slots=True)
class RequestContext:
    tenant_id: int
    request_id: str
    trace_id: str
    client_ip: str
    actor: str


_rate_limiter: RateLimiter | None = None


async def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = get_redis_rate_limiter()
    return _rate_limiter


async def request_context(request: Request) -> RequestContext:
    tenant_id = await _resolve_tenant_id(request)
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    trace_id = request.headers.get("x-trace-id") or request_id
    client_ip = request.client.host if request.client is not None else "unknown"
    actor = "anonymous"
    return RequestContext(
        tenant_id=tenant_id,
        request_id=request_id,
        trace_id=trace_id,
        client_ip=client_ip,
        actor=actor,
    )


async def rate_limit_dependency(
    request: Request,
    context: Annotated[RequestContext, Depends(request_context)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> None:
    session_id = await _session_id(request)
    try:
        await limiter.check(
            session_id=session_id or f"tenant:{context.tenant_id}:anonymous",
            client_ip=context.client_ip,
        )
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc


async def _resolve_tenant_id(request: Request) -> int:
    payload = await _payload(request)
    qr_payload = _first_present(payload.get("qr_payload"), request.query_params.get("qr_payload"))
    if qr_payload is not None:
        try:
            return parse_tenant_context(qr_payload).tenant_id
        except InvalidQrPayloadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    tenant_id = _first_present(payload.get("tenant_id"), request.query_params.get("tenant_id"))
    if tenant_id is None:
        raise HTTPException(status_code=400, detail="tenant_id or qr_payload is required.")
    try:
        parsed_tenant_id = int(str(tenant_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="tenant_id must be a positive integer.",
        ) from exc
    if parsed_tenant_id <= 0:
        raise HTTPException(status_code=400, detail="tenant_id must be a positive integer.")
    return parsed_tenant_id


async def _session_id(request: Request) -> str | None:
    payload = await _payload(request)
    value = _first_present(payload.get("session_id"), request.query_params.get("session_id"))
    return str(value) if value is not None else None


async def _payload(request: Request) -> dict[str, Any]:
    if request.method in {"GET", "DELETE"}:
        return {}
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001 - body may be empty or not JSON.
        return {}
    return data if isinstance(data, dict) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
