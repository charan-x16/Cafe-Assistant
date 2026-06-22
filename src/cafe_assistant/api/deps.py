from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from cafe_assistant.db.models import Location
from cafe_assistant.db.session import get_session
from cafe_assistant.identity.qr import InvalidQrPayloadError, parse_tenant_context
from cafe_assistant.security.rate_limit import (
    RateLimiter,
    RateLimitExceededError,
    get_redis_rate_limiter,
)

SessionDependency = Annotated[AsyncSession, Depends(get_session)]
DEVICE_TOKEN_COOKIE_NAME = "cafe_assistant_device_token"


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Tenant and trace context resolved for one API request.

    Attributes:
        tenant_id (int):
            Tenant that owns all data accessed by the request.
        location_id (int | None):
            Location from a validated QR payload, or None for direct tenant requests.
        table_id (str | None):
            Table identifier from a validated QR payload, or None when absent.
        request_id (str):
            Request identifier propagated to traces and audit rows.
        trace_id (str):
            Trace identifier propagated across observability boundaries.
        client_ip (str):
            Best-effort client IP used by rate limiting.
        actor (str):
            Anonymous or customer actor label for audit events.
    """

    tenant_id: int
    location_id: int | None
    table_id: str | None
    request_id: str
    trace_id: str
    client_ip: str
    actor: str


@dataclass(frozen=True, slots=True)
class _ResolvedTenantContext:
    """Internal tenant context after request parsing and optional QR validation."""

    tenant_id: int
    location_id: int | None = None
    table_id: str | None = None


_rate_limiter: RateLimiter | None = None


async def get_rate_limiter() -> RateLimiter:
    """Return the process-wide Redis-backed rate limiter.

    Args:
        None:
            The limiter is built from environment settings on first use.

    Returns:
        RateLimiter:
            Cached limiter instance used by API dependencies.
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = get_redis_rate_limiter()
    return _rate_limiter


async def request_context(
    request: Request,
    session: SessionDependency,
) -> RequestContext:
    """Resolve tenant, QR, request, and trace context for an API call.

    Args:
        request (Request):
            Incoming FastAPI request containing body, query params, headers, and client info.
        session (AsyncSession):
            Database session used to validate QR tenant/location ownership.

    Returns:
        RequestContext:
            Fully resolved tenant-scoped request context. QR requests preserve
            location and table identifiers for downstream tracing and audit.
    """
    tenant_context = await _resolve_tenant_context(request, session)
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    trace_id = request.headers.get("x-trace-id") or request_id
    client_ip = request.client.host if request.client is not None else "unknown"
    actor = "anonymous"
    return RequestContext(
        tenant_id=tenant_context.tenant_id,
        location_id=tenant_context.location_id,
        table_id=tenant_context.table_id,
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
    """Apply per-tenant session and IP rate limits to an API request.

    Args:
        request (Request):
            Incoming request used to read an optional session ID from the payload.
        context (RequestContext):
            Tenant-scoped request context resolved before limiting.
        limiter (RateLimiter):
            Configured rate limiter implementation.

    Returns:
        None:
            The request is allowed to continue unless a limit is exceeded.
    """
    session_id = await _session_id(request)
    rate_session_id = (
        f"tenant:{context.tenant_id}:session:{session_id}"
        if session_id
        else f"tenant:{context.tenant_id}:anonymous"
    )
    try:
        await limiter.check(
            session_id=rate_session_id,
            client_ip=context.client_ip,
        )
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc


async def device_token_from_request(request: Request) -> str | None:
    """Extract a device token only from approved non-URL transports.

    Args:
        request (Request):
            Incoming FastAPI request that may carry a Bearer token header or
            secure-cookie token.

    Returns:
        str | None:
            Opaque device token when supplied through an approved transport;
            None for anonymous requests.
    """
    payload = await _payload(request)
    if request.query_params.get("device_token") is not None or payload.get("device_token") is not None:
        raise HTTPException(
            status_code=400,
            detail="device_token must be sent in the Authorization header or secure cookie.",
        )

    bearer_token = _bearer_token(request.headers.get("authorization"))
    cookie_token = request.cookies.get(DEVICE_TOKEN_COOKIE_NAME)
    if bearer_token and cookie_token and bearer_token != cookie_token:
        raise HTTPException(status_code=400, detail="Conflicting device tokens supplied.")
    return bearer_token or cookie_token


async def _resolve_tenant_context(
    request: Request,
    session: AsyncSession,
) -> _ResolvedTenantContext:
    """Resolve direct tenant or QR-derived tenant context from a request.

    Args:
        request (Request):
            Incoming request containing a JSON body or query parameters.
        session (AsyncSession):
            Database session used for QR location ownership validation.

    Returns:
        _ResolvedTenantContext:
            Tenant context, including location/table fields when QR supplied.
    """
    payload = await _payload(request)
    qr_payload = _first_present(payload.get("qr_payload"), request.query_params.get("qr_payload"))
    if qr_payload is not None:
        try:
            parsed = parse_tenant_context(qr_payload)
        except (InvalidQrPayloadError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await _validate_qr_location(session, tenant_id=parsed.tenant_id, location_id=parsed.location_id)
        return _ResolvedTenantContext(
            tenant_id=parsed.tenant_id,
            location_id=parsed.location_id,
            table_id=parsed.table_id,
        )

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
    return _ResolvedTenantContext(tenant_id=parsed_tenant_id)


async def _validate_qr_location(
    session: AsyncSession,
    *,
    tenant_id: int,
    location_id: int,
) -> None:
    """Validate that a QR location exists inside the supplied tenant.

    Args:
        session (AsyncSession):
            Database session used for the ownership check.
        tenant_id (int):
            Tenant parsed from the QR cafe_id field.
        location_id (int):
            Location parsed from the QR location_id field.

    Returns:
        None:
            The function returns only when the location belongs to the tenant.
    """
    found = await session.scalar(
        select(Location.id).where(Location.id == location_id, Location.tenant_id == tenant_id)
    )
    if found is None:
        raise HTTPException(status_code=400, detail="QR location does not belong to this tenant.")


async def _session_id(request: Request) -> str | None:
    """Read the optional session ID used by rate limiting.

    Args:
        request (Request):
            Incoming request body or query parameters.

    Returns:
        str | None:
            Session ID when supplied, otherwise None.
    """
    payload = await _payload(request)
    value = _first_present(payload.get("session_id"), request.query_params.get("session_id"))
    return str(value) if value is not None else None


async def _payload(request: Request) -> dict[str, Any]:
    """Return a JSON object body for methods that may carry one.

    Args:
        request (Request):
            Incoming request whose body may or may not be JSON.

    Returns:
        dict[str, Any]:
            Parsed JSON object body, or an empty dict for non-object or absent bodies.
    """
    if request.method in {"GET", "DELETE"}:
        return {}
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001 - body may be empty or not JSON.
        return {}
    return data if isinstance(data, dict) else {}


def _bearer_token(header_value: str | None) -> str | None:
    """Parse an Authorization Bearer token header.

    Args:
        header_value (str | None):
            Raw Authorization header value.

    Returns:
        str | None:
            Bearer token when present, otherwise None.
    """
    if header_value is None or not header_value.strip():
        return None
    scheme, _, token = header_value.strip().partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=400, detail="Authorization must use Bearer token syntax.")
    return token.strip()


def _first_present(*values: Any) -> Any:
    """Return the first value that is not None.

    Args:
        *values (Any):
            Candidate values in precedence order.

    Returns:
        Any:
            First non-None value, or None when every value is absent.
    """
    for value in values:
        if value is not None:
            return value
    return None
