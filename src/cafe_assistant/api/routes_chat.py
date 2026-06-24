from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.agent.state_machine import ChatAgent, ChatAgentRequest
from cafe_assistant.api.deps import (
    RequestContext,
    device_token_from_request,
    rate_limit_dependency,
    request_context,
)
from cafe_assistant.db.session import get_session
from cafe_assistant.observability.metrics import RequestTimer
from cafe_assistant.observability.tracing import finish_trace, start_trace

router = APIRouter()
logger = logging.getLogger("uvicorn.error")
_STATIC_DIR = Path(__file__).parents[3] / "static"


class ChatRequest(BaseModel):
    """Streaming chat request body without durable identity secrets.

    Attributes:
        session_id (str):
            Browser session key for short-lived memory.
        tenant_id (int | None):
            Direct tenant context for local/dev clients.
        qr_payload (str | dict[str, Any] | None):
            QR-derived cafe/location/table context.
        message (str):
            Customer message to route through the safety-first agent.
    """

    session_id: str = Field(min_length=1)
    tenant_id: int | None = None
    qr_payload: str | dict[str, Any] | None = None
    message: str = Field(min_length=1)


SessionDependency = Annotated[AsyncSession, Depends(get_session)]
RequestContextDependency = Annotated[RequestContext, Depends(request_context)]
RateLimitDependency = Annotated[None, Depends(rate_limit_dependency)]
DeviceTokenDependency = Annotated[str | None, Depends(device_token_from_request)]


@router.post("/chat")
async def chat(
    request: ChatRequest,
    session: SessionDependency,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
    device_token: DeviceTokenDependency,
) -> StreamingResponse:
    """Run the tenant-scoped chat state machine and stream SSE chunks.

    Args:
        request (ChatRequest):
            Session ID, tenant/QR context, and user message.
        session (AsyncSession):
            Database session used by the agent and deterministic tools.
        context (RequestContext):
            Resolved tenant, optional QR location/table, and trace metadata.
        _rate_limited (None):
            Dependency marker confirming rate-limit checks have run.
        device_token (str | None):
            Optional durable identity token from approved transports.

    Returns:
        StreamingResponse:
            Server-sent-event stream containing response token chunks.
    """
    del _rate_limited
    start_trace(
        tenant_id=context.tenant_id,
        request_id=context.request_id,
        trace_id=context.trace_id,
    )
    timer = RequestTimer("/chat")
    started_at = time.perf_counter()
    agent = ChatAgent(session)

    async def event_stream() -> AsyncIterator[str]:
        """Yield SSE events from the chat agent.

        Args:
            None:
                The closure captures the request, context, token, and agent.

        Returns:
            AsyncIterator[str]:
                SSE-formatted token and completion events.
        """
        ok = False
        try:
            async for token in agent.stream_response(
                ChatAgentRequest(
                    session_id=request.session_id,
                    tenant_id=context.tenant_id,
                    message=request.message,
                    device_token=device_token,
                    location_id=context.location_id,
                    table_id=context.table_id,
                    request_id=context.request_id,
                    trace_id=context.trace_id,
                    actor=context.actor,
                )
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            ok = True
            yield "event: done\ndata: {}\n\n"
        finally:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
            completed_at = datetime.now(UTC).isoformat()
            logger.info(
                "chat_response_completed completed_at=%s duration_ms=%s ok=%s "
                "tenant_id=%s request_id=%s trace_id=%s",
                completed_at,
                elapsed_ms,
                ok,
                context.tenant_id,
                context.request_id,
                context.trace_id,
            )
            timer.finish(ok=ok)
            finish_trace(context.trace_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "X-Request-ID": context.request_id,
            "X-Trace-ID": context.trace_id,
        },
    )


@router.get("/chat")
async def chat_page() -> FileResponse:
    """Return the minimal browser chat page.

    Args:
        None:
            The page path is fixed relative to the package root.

    Returns:
        FileResponse:
            Static HTML chat interface.
    """
    return FileResponse(_STATIC_DIR / "chat.html")
