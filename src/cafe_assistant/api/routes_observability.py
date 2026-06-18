from __future__ import annotations

from fastapi import APIRouter, HTTPException

from cafe_assistant.observability.metrics import get_metrics_registry
from cafe_assistant.observability.replay import (
    TraceNotFoundError,
)
from cafe_assistant.observability.replay import (
    replay_trace as replay_stored_trace,
)

router = APIRouter(tags=["observability"])


@router.get("/metrics")
async def metrics() -> dict[str, object]:
    return get_metrics_registry().snapshot()


@router.get("/observability/replay/{trace_id}")
async def replay_trace(trace_id: str) -> dict[str, object]:
    try:
        return replay_stored_trace(trace_id)
    except TraceNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Trace not found.") from exc
