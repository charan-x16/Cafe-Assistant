from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from cafe_assistant.agent.state_machine import ChatAgent, ChatAgentRequest
from cafe_assistant.api.deps import (
    RequestContext,
    rate_limit_dependency,
    request_context,
)
from cafe_assistant.db.session import get_session
from cafe_assistant.observability.metrics import RequestTimer
from cafe_assistant.observability.tracing import start_trace

router = APIRouter()
_STATIC_DIR = Path(__file__).parents[3] / "static"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    tenant_id: int | None = None
    qr_payload: str | dict[str, Any] | None = None
    device_token: str | None = None
    message: str = Field(min_length=1)


SessionDependency = Annotated[AsyncSession, Depends(get_session)]
RequestContextDependency = Annotated[RequestContext, Depends(request_context)]
RateLimitDependency = Annotated[None, Depends(rate_limit_dependency)]


@router.post("/chat")
async def chat(
    request: ChatRequest,
    session: SessionDependency,
    context: RequestContextDependency,
    _rate_limited: RateLimitDependency,
) -> StreamingResponse:
    del _rate_limited
    start_trace(
        tenant_id=context.tenant_id,
        request_id=context.request_id,
        trace_id=context.trace_id,
    )
    timer = RequestTimer("/chat")
    agent = ChatAgent(session)

    async def event_stream() -> AsyncIterator[str]:
        ok = False
        try:
            async for token in agent.stream_response(
                ChatAgentRequest(
                    session_id=request.session_id,
                    tenant_id=context.tenant_id,
                    message=request.message,
                    device_token=request.device_token,
                    request_id=context.request_id,
                    trace_id=context.trace_id,
                    actor=context.actor,
                )
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            ok = True
            yield "event: done\ndata: {}\n\n"
        finally:
            timer.finish(ok=ok)

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
    return FileResponse(_STATIC_DIR / "chat.html")
