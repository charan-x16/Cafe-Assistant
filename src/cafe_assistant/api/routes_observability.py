"""Protected observability endpoints for metrics and incident replay.

Metrics and replay data can reveal operational behavior, model context, and trace
metadata. These routes therefore require a valid tenant context, normal API rate
limits, and an explicit admin token before returning any data. Trace replay is
further constrained to the tenant on the request.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from cafe_assistant.api.deps import (
    RequestContext,
    admin_auth_dependency,
    rate_limit_dependency,
    request_context,
)
from cafe_assistant.observability.metrics import get_metrics_registry
from cafe_assistant.observability.replay import TraceNotFoundError
from cafe_assistant.observability.replay import replay_trace as replay_stored_trace

router = APIRouter(tags=["observability"])
RequestContextDependency = Annotated[RequestContext, Depends(request_context)]
RateLimitDependency = Annotated[None, Depends(rate_limit_dependency)]
AdminDependency = Annotated[None, Depends(admin_auth_dependency)]


@router.get("/metrics")
async def metrics(
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
    _admin: AdminDependency,
) -> dict[str, object]:
    """Return in-process metrics to an authorized tenant-scoped admin caller.

    Args:
        context (RequestContext):
            Tenant, request, and trace context resolved for the metrics request.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.
        _admin (None):
            Dependency marker confirming admin authentication has passed.

    Returns:
        dict[str, object]:
            Current metrics snapshot grouped by reliability, quality, latency, and cost.
    """
    del context, _rate_limited, _admin
    return get_metrics_registry().snapshot()


@router.get("/metrics/openmetrics", response_class=PlainTextResponse)
async def openmetrics(
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
    _admin: AdminDependency,
) -> PlainTextResponse:
    """Return OpenMetrics text to an authorized tenant-scoped admin caller.

    Args:
        context (RequestContext):
            Tenant, request, and trace context resolved for the metrics request.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.
        _admin (None):
            Dependency marker confirming admin authentication has passed.

    Returns:
        PlainTextResponse:
            OpenMetrics-compatible text payload suitable for scraping.
    """
    del context, _rate_limited, _admin
    return PlainTextResponse(
        get_metrics_registry().to_openmetrics(),
        media_type="application/openmetrics-text; version=1.0.0; charset=utf-8",
    )


@router.get("/observability/replay/{trace_id}")
async def replay_trace(
    trace_id: str,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
    _admin: AdminDependency,
) -> dict[str, object]:
    """Replay a stored trace only when it belongs to the request tenant.

    Args:
        trace_id (str):
            Trace identifier from the request path or audit record.
        context (RequestContext):
            Tenant context used to enforce replay isolation.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.
        _admin (None):
            Dependency marker confirming admin authentication has passed.

    Returns:
        dict[str, object]:
            Redacted trace replay payload for incident debugging.
    """
    del _rate_limited, _admin
    try:
        return replay_stored_trace(trace_id, tenant_id=context.tenant_id)
    except TraceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Trace not found.") from exc
