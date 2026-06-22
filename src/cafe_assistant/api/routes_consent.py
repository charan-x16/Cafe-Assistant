"""Identity, consent, and profile inspection routes.

These endpoints support the anonymous-to-remembered customer upgrade flow. OTP
confirmation can attach consent-gated health facts from tenant-scoped session
memory, profile reads are available only through a tenant-matched device token
sent by approved transports, and deletion removes durable customer memory
through repository cascade paths.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.api.deps import (
    DEVICE_TOKEN_COOKIE_NAME,
    RequestContext,
    device_token_from_request,
    rate_limit_dependency,
    request_context,
)
from cafe_assistant.config import settings
from cafe_assistant.db.repositories.consent_repo import (
    DIETARY_HEALTH_SCOPE,
    InvalidConsentScopeError,
)
from cafe_assistant.db.repositories.profile_repo import delete_customer_profile, load_stored_profile
from cafe_assistant.db.session import get_session
from cafe_assistant.identity.device import verify_device_token
from cafe_assistant.identity.otp import OtpError, OtpService, get_default_otp_service
from cafe_assistant.memory.session import get_redis_session_memory
from cafe_assistant.security.audit import AuditContext, append_audit_event

router = APIRouter(prefix="/identity", tags=["identity"])
SessionDependency = Annotated[AsyncSession, Depends(get_session)]
RequestContextDependency = Annotated[RequestContext, Depends(request_context)]
RateLimitDependency = Annotated[None, Depends(rate_limit_dependency)]
DeviceTokenDependency = Annotated[str | None, Depends(device_token_from_request)]
OtpServiceDependency = Annotated[OtpService, Depends(get_default_otp_service)]


class TenantContextRequest(BaseModel):
    """Base request body carrying explicit or QR-derived tenant context."""

    tenant_id: int | None = None
    qr_payload: str | dict[str, Any] | None = None


class OtpStartRequest(TenantContextRequest):
    """Request body for starting an OTP consent upgrade."""

    phone: str = Field(min_length=7)
    scopes: list[str] = Field(default_factory=lambda: [DIETARY_HEALTH_SCOPE])


class OtpStartResponse(BaseModel):
    """Response returned after an OTP challenge is created."""

    challenge_id: str
    expires_at: datetime


class OtpConfirmRequest(TenantContextRequest):
    """Request body for confirming an OTP challenge and linking a profile."""

    session_id: str | None = None
    phone: str = Field(min_length=7)
    challenge_id: str = Field(min_length=1)
    code: str = Field(min_length=4, max_length=12)


class OtpConfirmResponse(BaseModel):
    """Response returned after a successful OTP profile upgrade."""

    customer_id: int
    tenant_id: int
    device_token: str
    granted_scopes: list[str]


class ProfileResponse(BaseModel):
    """Inspectable durable profile returned to a recognized customer."""

    customer_id: int
    tenant_id: int
    preferences: dict[str, Any]
    dietary_facts: dict[str, Any]
    consent_at: datetime | None
    recent_events: list[dict[str, Any]]


class DeleteProfileResponse(BaseModel):
    """Response returned after a right-to-erasure request."""

    deleted: bool


@router.post("/otp/start")
async def start_otp(
    request: OtpStartRequest,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
    otp_service: OtpServiceDependency,
) -> OtpStartResponse:
    """Start an OTP challenge for a tenant-scoped consent upgrade.

    Args:
        request (OtpStartRequest):
            Phone number, requested scopes, and tenant context supplied by the client.
        context (RequestContext):
            Resolved tenant, optional QR location/table, and request metadata.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.
        otp_service (OtpService):
            OTP service dependency used to create and send the challenge.

    Returns:
        OtpStartResponse:
            Challenge identifier and expiration time for the OTP code.
    """
    del _rate_limited
    try:
        result = await otp_service.start(
            tenant_id=context.tenant_id,
            phone=request.phone,
            scopes=tuple(request.scopes),
        )
    except (OtpError, InvalidConsentScopeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OtpStartResponse(challenge_id=result.challenge_id, expires_at=result.expires_at)


@router.post("/otp/confirm")
async def confirm_otp(
    request: OtpConfirmRequest,
    response: Response,
    session: SessionDependency,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
    otp_service: OtpServiceDependency,
) -> OtpConfirmResponse:
    """Confirm an OTP challenge and create or link durable customer identity.

    Args:
        request (OtpConfirmRequest):
            OTP code, phone number, optional session ID, and tenant context.
        response (Response):
            FastAPI response used to set the approved HttpOnly device-token cookie.
        session (AsyncSession):
            Async database session used for profile, consent, token, and audit writes.
        context (RequestContext):
            Resolved tenant, optional QR location/table, and request metadata.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.
        otp_service (OtpService):
            OTP service dependency used to confirm the challenge.

    Returns:
        OtpConfirmResponse:
            Durable customer ID, tenant ID, opaque device token, and granted scopes.
    """
    del _rate_limited
    session_state = None
    if request.session_id:
        try:
            session_state = await get_redis_session_memory().load(
                tenant_id=context.tenant_id,
                session_id=request.session_id,
            )
        except Exception:
            session_state = None

    try:
        result = await otp_service.confirm(
            session,
            tenant_id=context.tenant_id,
            phone=request.phone,
            challenge_id=request.challenge_id,
            code=request.code,
            session_state=session_state,
        )
    except OtpError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await append_audit_event(
        session,
        context=_audit_context(context, actor=f"customer:{result.customer_id}"),
        action="consent_granted",
        payload={
            "phone": request.phone,
            "scopes": list(result.granted_scopes),
            "customer_id": result.customer_id,
            "location_id": context.location_id,
            "table_id": context.table_id,
        },
    )
    await append_audit_event(
        session,
        context=_audit_context(context, actor=f"customer:{result.customer_id}"),
        action="profile_write",
        payload={
            "source": "otp_confirm",
            "session_id": request.session_id,
            "customer_id": result.customer_id,
            "location_id": context.location_id,
            "table_id": context.table_id,
        },
    )
    response.set_cookie(
        DEVICE_TOKEN_COOKIE_NAME,
        result.device_token,
        httponly=True,
        secure=settings.environment.strip().lower() not in {"local", "test", "development"},
        samesite="lax",
        max_age=settings.device_token_ttl_seconds,
    )
    return OtpConfirmResponse(
        customer_id=result.customer_id,
        tenant_id=result.tenant_id,
        device_token=result.device_token,
        granted_scopes=list(result.granted_scopes),
    )


@router.get("/profile")
async def get_profile(
    session: SessionDependency,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
    device_token: DeviceTokenDependency,
    tenant_id: int | None = Query(default=None),
    qr_payload: str | None = Query(default=None),
) -> ProfileResponse:
    """Return an inspectable profile for a recognized tenant-scoped device token.

    Args:
        session (AsyncSession):
            Async database session used for profile read and audit write.
        context (RequestContext):
            Resolved tenant, optional QR location/table, and request metadata.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.
        device_token (str | None):
            Opaque browser token from Authorization header or secure cookie.
        tenant_id (int | None):
            Query tenant ID consumed by request-context resolution.
        qr_payload (str | None):
            Query QR payload consumed by request-context resolution.

    Returns:
        ProfileResponse:
            Stored preferences, dietary facts, consent timestamp, and recent events.
    """
    del tenant_id, qr_payload, _rate_limited
    identity = await verify_device_token(
        session,
        tenant_id=context.tenant_id,
        token=device_token,
    )
    if identity is None:
        raise HTTPException(status_code=404, detail="No recognized profile.")

    profile = await load_stored_profile(
        session,
        tenant_id=context.tenant_id,
        customer_id=identity.customer_id,
    )
    if profile is None:
        raise HTTPException(status_code=404, detail="No recognized profile.")

    await append_audit_event(
        session,
        context=_audit_context(context, actor=f"customer:{identity.customer_id}"),
        action="profile_read",
        payload={
            "customer_id": identity.customer_id,
            "location_id": context.location_id,
            "table_id": context.table_id,
        },
    )
    return ProfileResponse(
        customer_id=profile.customer_id,
        tenant_id=profile.tenant_id,
        preferences=profile.preferences,
        dietary_facts=profile.dietary_facts,
        consent_at=profile.consent_at,
        recent_events=[
            {
                "type": event.type,
                "payload": event.payload,
                "created_at": event.created_at,
            }
            for event in profile.recent_events
        ],
    )


@router.delete("/profile")
async def delete_profile(
    session: SessionDependency,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
    device_token: DeviceTokenDependency,
    tenant_id: int | None = Query(default=None),
    qr_payload: str | None = Query(default=None),
) -> DeleteProfileResponse:
    """Delete a recognized customer's tenant-scoped durable profile.

    Args:
        session (AsyncSession):
            Async database session used for deletion and audit write.
        context (RequestContext):
            Resolved tenant, optional QR location/table, and request metadata.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.
        device_token (str | None):
            Opaque browser token from Authorization header or secure cookie.
        tenant_id (int | None):
            Query tenant ID consumed by request-context resolution.
        qr_payload (str | None):
            Query QR payload consumed by request-context resolution.

    Returns:
        DeleteProfileResponse:
            Whether a tenant-scoped profile was found and deleted.
    """
    del tenant_id, qr_payload, _rate_limited
    identity = await verify_device_token(
        session,
        tenant_id=context.tenant_id,
        token=device_token,
    )
    if identity is None:
        return DeleteProfileResponse(deleted=False)

    deleted = await delete_customer_profile(
        session,
        tenant_id=context.tenant_id,
        customer_id=identity.customer_id,
    )
    if deleted:
        await append_audit_event(
            session,
            context=_audit_context(context, actor=f"customer:{identity.customer_id}"),
            action="profile_deleted",
            payload={
                "customer_id": identity.customer_id,
                "location_id": context.location_id,
                "table_id": context.table_id,
            },
        )
    return DeleteProfileResponse(deleted=deleted)


def _audit_context(context: RequestContext, *, actor: str | None = None) -> AuditContext:
    """Build audit context for identity/profile route events.

    Args:
        context (RequestContext):
            Resolved tenant, request, trace, IP, and actor metadata.
        actor (str | None):
            Optional actor override for recognized customer events.

    Returns:
        AuditContext:
            Tenant-scoped audit context passed to append-only audit logging.
    """
    return AuditContext(
        tenant_id=context.tenant_id,
        actor=actor or context.actor,
        request_id=context.request_id,
        trace_id=context.trace_id,
    )
