"""Account identity, consent, and profile inspection routes.

These endpoints provide tenant-scoped
username/password accounts. Registering or logging in returns an opaque bearer
token that identifies a customer profile for the current cafe. Preferences may
be copied from the anonymous session automatically, while health and dietary
facts are copied only through an explicit consent endpoint.
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
from cafe_assistant.db.repositories.consent_repo import DIETARY_HEALTH_SCOPE, grant_consent
from cafe_assistant.db.repositories.profile_repo import (
    append_event,
    delete_customer_profile,
    load_stored_profile,
    update_dietary_facts,
    update_preferences,
)
from cafe_assistant.db.session import get_session
from cafe_assistant.identity.account import (
    AccountIdentityError,
    authenticate_customer_account,
    create_customer_account,
)
from cafe_assistant.identity.device import (
    issue_device_token,
    revoke_device_token,
    verify_device_token,
)
from cafe_assistant.identity.dietary_facts import restrictions_to_dietary_facts
from cafe_assistant.memory.session import get_redis_session_memory
from cafe_assistant.security.audit import AuditContext, append_audit_event

router = APIRouter(prefix="/identity", tags=["identity"])
SessionDependency = Annotated[AsyncSession, Depends(get_session)]
RequestContextDependency = Annotated[RequestContext, Depends(request_context)]
RateLimitDependency = Annotated[None, Depends(rate_limit_dependency)]
AuthTokenDependency = Annotated[str | None, Depends(device_token_from_request)]


class TenantContextRequest(BaseModel):
    """Base request body carrying explicit or QR-derived tenant context.

    Attributes:
        tenant_id (int | None):
            Direct tenant ID for local/dev clients.
        qr_payload (str | dict[str, Any] | None):
            Optional QR payload containing cafe, location, and table context.
    """

    tenant_id: int | None = None
    qr_payload: str | dict[str, Any] | None = None


class AccountRequest(TenantContextRequest):
    """Request body for registering or logging in with an account.

    Attributes:
        username (str):
            Tenant-scoped username or email-like identifier.
        password (str):
            Raw password sent over HTTPS and hashed by the server.
        session_id (str | None):
            Optional anonymous session ID whose non-health preferences should be
            copied into the durable profile after authentication.
    """

    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8, max_length=128)
    session_id: str | None = None


class AuthResponse(BaseModel):
    """Response returned after successful registration or login.

    Attributes:
        customer_id (int):
            Durable customer profile ID inside the tenant.
        tenant_id (int):
            Tenant that owns the account and profile.
        username (str):
            Normalized username stored for the account.
        auth_token (str):
            Opaque bearer token returned once for browser storage.
    """

    customer_id: int
    tenant_id: int
    username: str
    auth_token: str


class LogoutResponse(BaseModel):
    """Response returned after revoking the current auth token."""

    logged_out: bool


class HealthConsentRequest(TenantContextRequest):
    """Request body for explicitly remembering health/dietary facts.

    Attributes:
        session_id (str | None):
            Optional anonymous/current session whose restrictions should be
            copied into the durable profile after consent is granted.
    """

    session_id: str | None = None


class HealthConsentResponse(BaseModel):
    """Response returned after granting health-data consent."""

    granted: bool
    dietary_facts_saved: bool


class ProfileResponse(BaseModel):
    """Inspectable durable profile returned to a recognized customer."""

    customer_id: int
    tenant_id: int
    username: str | None = None
    preferences: dict[str, Any]
    dietary_facts: dict[str, Any]
    consent_at: datetime | None
    recent_events: list[dict[str, Any]]


class DeleteProfileResponse(BaseModel):
    """Response returned after a right-to-erasure request."""

    deleted: bool


@router.post("/register")
async def register(
    request: AccountRequest,
    response: Response,
    session: SessionDependency,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
) -> AuthResponse:
    """Create a username/password account for the current tenant.

    Args:
        request (AccountRequest):
            Account credentials, tenant context, and optional session ID.
        response (Response):
            FastAPI response used to set the HttpOnly auth-token cookie.
        session (AsyncSession):
            Async database session used for account, token, profile, and audit writes.
        context (RequestContext):
            Resolved tenant, location/table, request, and trace context.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.

    Returns:
        AuthResponse:
            Durable customer identity and opaque auth token.
    """
    del _rate_limited
    try:
        account = await create_customer_account(
            session,
            tenant_id=context.tenant_id,
            username=request.username,
            password=request.password,
        )
    except AccountIdentityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _copy_session_preferences(
        session,
        tenant_id=context.tenant_id,
        customer_id=account.customer_id,
        session_id=request.session_id,
    )
    token = await issue_device_token(
        session,
        tenant_id=context.tenant_id,
        customer_id=account.customer_id,
    )
    _set_auth_cookie(response, token)
    await append_audit_event(
        session,
        context=_audit_context(context, actor=f"customer:{account.customer_id}"),
        action="account_registered",
        payload={"customer_id": account.customer_id, "username": account.username},
    )
    return AuthResponse(
        customer_id=account.customer_id,
        tenant_id=account.tenant_id,
        username=account.username,
        auth_token=token,
    )


@router.post("/login")
async def login(
    request: AccountRequest,
    response: Response,
    session: SessionDependency,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
) -> AuthResponse:
    """Authenticate an existing tenant-scoped account.

    Args:
        request (AccountRequest):
            Account credentials, tenant context, and optional session ID.
        response (Response):
            FastAPI response used to set the HttpOnly auth-token cookie.
        session (AsyncSession):
            Async database session used for account, token, profile, and audit writes.
        context (RequestContext):
            Resolved tenant, location/table, request, and trace context.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.

    Returns:
        AuthResponse:
            Durable customer identity and opaque auth token.
    """
    del _rate_limited
    try:
        account = await authenticate_customer_account(
            session,
            tenant_id=context.tenant_id,
            username=request.username,
            password=request.password,
        )
    except AccountIdentityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if account is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    await _copy_session_preferences(
        session,
        tenant_id=context.tenant_id,
        customer_id=account.customer_id,
        session_id=request.session_id,
    )
    token = await issue_device_token(
        session,
        tenant_id=context.tenant_id,
        customer_id=account.customer_id,
    )
    _set_auth_cookie(response, token)
    await append_audit_event(
        session,
        context=_audit_context(context, actor=f"customer:{account.customer_id}"),
        action="account_login",
        payload={"customer_id": account.customer_id, "username": account.username},
    )
    return AuthResponse(
        customer_id=account.customer_id,
        tenant_id=account.tenant_id,
        username=account.username,
        auth_token=token,
    )


@router.post("/logout")
async def logout(
    response: Response,
    session: SessionDependency,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
    auth_token: AuthTokenDependency,
    tenant_id: int | None = Query(default=None),
    qr_payload: str | None = Query(default=None),
) -> LogoutResponse:
    """Revoke the current auth token without deleting profile data.

    Args:
        response (Response):
            FastAPI response used to clear the browser-held auth cookie.
        session (AsyncSession):
            Async database session used for token revocation and audit writes.
        context (RequestContext):
            Resolved tenant, location/table, request, and trace context.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.
        auth_token (str | None):
            Bearer or cookie token identifying the current account session.
        tenant_id (int | None):
            Query tenant ID consumed by request-context resolution.
        qr_payload (str | None):
            Query QR payload consumed by request-context resolution.

    Returns:
        LogoutResponse:
            Whether an active token was revoked.
    """
    del tenant_id, qr_payload, _rate_limited
    response.delete_cookie(DEVICE_TOKEN_COOKIE_NAME)
    if not auth_token:
        return LogoutResponse(logged_out=False)
    identity = await verify_device_token(session, tenant_id=context.tenant_id, token=auth_token)
    revoked = await revoke_device_token(session, tenant_id=context.tenant_id, token=auth_token)
    if identity is not None:
        await append_audit_event(
            session,
            context=_audit_context(context, actor=f"customer:{identity.customer_id}"),
            action="account_logout",
            payload={"customer_id": identity.customer_id, "revoked": revoked},
        )
    return LogoutResponse(logged_out=revoked)


@router.post("/consent/health")
async def grant_health_consent(
    request: HealthConsentRequest,
    session: SessionDependency,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
    auth_token: AuthTokenDependency,
) -> HealthConsentResponse:
    """Grant consent to remember health/dietary facts for the account.

    Args:
        request (HealthConsentRequest):
            Tenant context and optional session ID containing current restrictions.
        session (AsyncSession):
            Async database session used for consent and profile writes.
        context (RequestContext):
            Resolved tenant, location/table, request, and trace context.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.
        auth_token (str | None):
            Bearer or cookie token identifying the logged-in customer.

    Returns:
        HealthConsentResponse:
            Whether consent was granted and whether session health facts were copied.
    """
    del _rate_limited
    identity = await verify_device_token(session, tenant_id=context.tenant_id, token=auth_token)
    if identity is None:
        raise HTTPException(status_code=401, detail="Login is required.")
    granted = await grant_consent(
        session,
        tenant_id=context.tenant_id,
        customer_id=identity.customer_id,
        scope=DIETARY_HEALTH_SCOPE,
    )
    dietary_facts_saved = False
    if request.session_id:
        try:
            state = await get_redis_session_memory().load(
                tenant_id=context.tenant_id,
                session_id=request.session_id,
            )
            facts = restrictions_to_dietary_facts(state.restrictions)
        except Exception:
            facts = {}
        if facts:
            dietary_facts_saved = await update_dietary_facts(
                session,
                tenant_id=context.tenant_id,
                customer_id=identity.customer_id,
                updates=facts,
            )
            if dietary_facts_saved:
                await append_event(
                    session,
                    tenant_id=context.tenant_id,
                    customer_id=identity.customer_id,
                    event_type="dietary_facts_saved",
                    payload={"keys": sorted(facts)},
                )
    await append_audit_event(
        session,
        context=_audit_context(context, actor=f"customer:{identity.customer_id}"),
        action="consent_granted",
        payload={
            "customer_id": identity.customer_id,
            "scope": DIETARY_HEALTH_SCOPE,
            "dietary_facts_saved": dietary_facts_saved,
        },
    )
    return HealthConsentResponse(granted=granted, dietary_facts_saved=dietary_facts_saved)


@router.get("/profile")
async def get_profile(
    session: SessionDependency,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
    auth_token: AuthTokenDependency,
    tenant_id: int | None = Query(default=None),
    qr_payload: str | None = Query(default=None),
) -> ProfileResponse:
    """Return an inspectable profile for a logged-in tenant-scoped account.

    Args:
        session (AsyncSession):
            Async database session used for profile read and audit write.
        context (RequestContext):
            Resolved tenant, optional QR location/table, and request metadata.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.
        auth_token (str | None):
            Opaque auth token from Authorization header or secure cookie.
        tenant_id (int | None):
            Query tenant ID consumed by request-context resolution.
        qr_payload (str | None):
            Query QR payload consumed by request-context resolution.

    Returns:
        ProfileResponse:
            Stored preferences, dietary facts, consent timestamp, and recent events.
    """
    del tenant_id, qr_payload, _rate_limited
    identity = await verify_device_token(session, tenant_id=context.tenant_id, token=auth_token)
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
    response: Response,
    session: SessionDependency,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
    auth_token: AuthTokenDependency,
    tenant_id: int | None = Query(default=None),
    qr_payload: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
) -> DeleteProfileResponse:
    """Delete a logged-in customer's tenant-scoped durable profile.

    Args:
        response (Response):
            FastAPI response used to clear the browser-held auth cookie.
        session (AsyncSession):
            Async database session used for deletion and audit write.
        context (RequestContext):
            Resolved tenant, optional QR location/table, and request metadata.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.
        auth_token (str | None):
            Opaque auth token from Authorization header or secure cookie.
        tenant_id (int | None):
            Query tenant ID consumed by request-context resolution.
        qr_payload (str | None):
            Query QR payload consumed by request-context resolution.
        session_id (str | None):
            Optional session key to purge from tenant-scoped Redis memory.

    Returns:
        DeleteProfileResponse:
            Whether a tenant-scoped profile was found and deleted.
    """
    del tenant_id, qr_payload, _rate_limited
    response.delete_cookie(DEVICE_TOKEN_COOKIE_NAME)
    identity = await verify_device_token(session, tenant_id=context.tenant_id, token=auth_token)
    if identity is None:
        return DeleteProfileResponse(deleted=False)

    deleted = await delete_customer_profile(
        session,
        tenant_id=context.tenant_id,
        customer_id=identity.customer_id,
    )
    session_memory_deleted = False
    if deleted and session_id:
        try:
            await get_redis_session_memory().delete(
                tenant_id=context.tenant_id,
                session_id=session_id,
            )
            session_memory_deleted = True
        except Exception:
            session_memory_deleted = False
    if deleted:
        await append_audit_event(
            session,
            context=_audit_context(context, actor=f"customer:{identity.customer_id}"),
            action="profile_deleted",
            payload={
                "customer_id": identity.customer_id,
                "session_memory_deleted": session_memory_deleted,
            },
        )
    return DeleteProfileResponse(deleted=deleted)


async def _copy_session_preferences(
    session: AsyncSession,
    *,
    tenant_id: int,
    customer_id: int,
    session_id: str | None,
) -> bool:
    """Copy anonymous non-health preferences into a logged-in profile.

    Args:
        session (AsyncSession):
            Async database session used for profile writes.
        tenant_id (int):
            Tenant that owns the profile and session memory.
        customer_id (int):
            Durable customer ID receiving preference updates.
        session_id (str | None):
            Optional anonymous/current session key.

    Returns:
        bool:
            True when preferences were found and copied, otherwise False.
    """
    if not session_id:
        return False
    try:
        state = await get_redis_session_memory().load(tenant_id=tenant_id, session_id=session_id)
    except Exception:
        return False
    if not state.preferences:
        return False
    saved = await update_preferences(
        session,
        tenant_id=tenant_id,
        customer_id=customer_id,
        updates=dict(state.preferences),
    )
    if saved:
        await append_event(
            session,
            tenant_id=tenant_id,
            customer_id=customer_id,
            event_type="preference_saved",
            payload={"keys": sorted(state.preferences)},
        )
    return saved


def _set_auth_cookie(response: Response, token: str) -> None:
    """Set the HttpOnly account auth cookie for browser clients.

    Args:
        response (Response):
            FastAPI response receiving the cookie header.
        token (str):
            Opaque auth token to store in the cookie.

    Returns:
        None:
            The response is mutated with a cookie header.
    """
    response.set_cookie(
        DEVICE_TOKEN_COOKIE_NAME,
        token,
        httponly=True,
        secure=settings.environment.strip().lower() not in {"local", "test", "development"},
        samesite="lax",
        max_age=settings.device_token_ttl_seconds,
    )


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