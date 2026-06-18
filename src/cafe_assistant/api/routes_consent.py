from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.api.deps import RequestContext, rate_limit_dependency, request_context
from cafe_assistant.db.repositories.consent_repo import DIETARY_HEALTH_SCOPE
from cafe_assistant.db.repositories.profile_repo import delete_customer_profile, load_stored_profile
from cafe_assistant.db.session import get_session
from cafe_assistant.identity.device import verify_device_token
from cafe_assistant.identity.otp import OtpError, OtpService
from cafe_assistant.memory.session import get_redis_session_memory
from cafe_assistant.security.audit import AuditContext, append_audit_event

router = APIRouter(prefix="/identity", tags=["identity"])
_otp_service = OtpService()
SessionDependency = Annotated[AsyncSession, Depends(get_session)]
RequestContextDependency = Annotated[RequestContext, Depends(request_context)]
RateLimitDependency = Annotated[None, Depends(rate_limit_dependency)]


class TenantContextRequest(BaseModel):
    tenant_id: int | None = None
    qr_payload: str | dict[str, Any] | None = None


class OtpStartRequest(TenantContextRequest):
    phone: str = Field(min_length=7)
    scopes: list[str] = Field(default_factory=lambda: [DIETARY_HEALTH_SCOPE])


class OtpStartResponse(BaseModel):
    challenge_id: str
    expires_at: datetime


class OtpConfirmRequest(TenantContextRequest):
    session_id: str | None = None
    phone: str = Field(min_length=7)
    challenge_id: str = Field(min_length=1)
    code: str = Field(min_length=4, max_length=12)


class OtpConfirmResponse(BaseModel):
    customer_id: int
    tenant_id: int
    device_token: str
    granted_scopes: list[str]


class ProfileResponse(BaseModel):
    customer_id: int
    tenant_id: int
    preferences: dict[str, Any]
    dietary_facts: dict[str, Any]
    consent_at: datetime | None
    recent_events: list[dict[str, Any]]


class DeleteProfileResponse(BaseModel):
    deleted: bool


@router.post("/otp/start")
async def start_otp(
    request: OtpStartRequest,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
) -> OtpStartResponse:
    del _rate_limited
    result = await _otp_service.start(
        tenant_id=context.tenant_id,
        phone=request.phone,
        scopes=tuple(request.scopes),
    )
    return OtpStartResponse(challenge_id=result.challenge_id, expires_at=result.expires_at)


@router.post("/otp/confirm")
async def confirm_otp(
    request: OtpConfirmRequest,
    session: SessionDependency,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
) -> OtpConfirmResponse:
    del _rate_limited
    session_state = None
    if request.session_id:
        session_state = await get_redis_session_memory().load(request.session_id)

    try:
        result = await _otp_service.confirm(
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
        },
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
    tenant_id: int | None = Query(default=None),
    qr_payload: str | None = Query(default=None),
    device_token: str | None = Query(default=None),
) -> ProfileResponse:
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
        payload={"customer_id": identity.customer_id},
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
    tenant_id: int | None = Query(default=None),
    qr_payload: str | None = Query(default=None),
    device_token: str | None = Query(default=None),
) -> DeleteProfileResponse:
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
            payload={"customer_id": identity.customer_id},
        )
    return DeleteProfileResponse(deleted=deleted)


def _audit_context(context: RequestContext, *, actor: str | None = None) -> AuditContext:
    return AuditContext(
        tenant_id=context.tenant_id,
        actor=actor or context.actor,
        request_id=context.request_id,
        trace_id=context.trace_id,
    )
