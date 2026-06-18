from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from cafe_assistant.config import settings
from cafe_assistant.security.redaction import redact_payload
from cafe_assistant.versioning import get_version_registry

_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cafe_assistant_trace_id",
    default=None,
)
_current_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cafe_assistant_request_id",
    default=None,
)


@dataclass(slots=True)
class SpanRecord:
    trace_id: str
    request_id: str
    name: str
    attributes: dict[str, Any]
    started_at: float
    ended_at: float | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000.0


@dataclass(slots=True)
class TraceRecord:
    trace_id: str
    request_id: str
    tenant_id: int
    version_registry: dict[str, object]
    spans: list[SpanRecord] = field(default_factory=list)


class TraceStore:
    def __init__(self, *, max_traces: int = 500) -> None:
        self.max_traces = max_traces
        self._traces: dict[str, TraceRecord] = {}
        self._order: list[str] = []

    def start_trace(self, *, trace_id: str, request_id: str, tenant_id: int) -> TraceRecord:
        record = TraceRecord(
            trace_id=trace_id,
            request_id=request_id,
            tenant_id=tenant_id,
            version_registry=get_version_registry().as_trace_attributes(),
        )
        self._traces[trace_id] = record
        self._order.append(trace_id)
        while len(self._order) > self.max_traces:
            oldest = self._order.pop(0)
            self._traces.pop(oldest, None)
        return record

    def add_span(self, span: SpanRecord) -> None:
        trace = self._traces.get(span.trace_id)
        if trace is not None:
            trace.spans.append(span)

    def get(self, trace_id: str) -> TraceRecord | None:
        return self._traces.get(trace_id)


class LangfuseClient:
    def __init__(self) -> None:
        self._client: object | None = None
        if not settings.langfuse_enabled:
            return
        try:
            from langfuse import Langfuse  # type: ignore[import-not-found]

            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception:
            self._client = None

    def emit_span(self, span: SpanRecord) -> None:
        if self._client is None:
            return
        try:
            trace = self._client.trace(  # type: ignore[attr-defined]
                id=span.trace_id,
                name="cafe_assistant_request",
                metadata={"request_id": span.request_id},
            )
            trace.span(
                name=span.name,
                metadata=span.attributes,
                start_time=span.started_at,
                end_time=span.ended_at,
            )
        except Exception:
            return


_trace_store = TraceStore()
_langfuse_client = LangfuseClient()


def get_trace_store() -> TraceStore:
    return _trace_store


def start_trace(*, tenant_id: int, request_id: str, trace_id: str | None = None) -> str:
    resolved_trace_id = trace_id or str(uuid.uuid4())
    _current_trace_id.set(resolved_trace_id)
    _current_request_id.set(request_id)
    _trace_store.start_trace(
        trace_id=resolved_trace_id,
        request_id=request_id,
        tenant_id=tenant_id,
    )
    return resolved_trace_id


def current_trace_id() -> str | None:
    return _current_trace_id.get()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[SpanRecord]:
    trace_id = _current_trace_id.get() or str(uuid.uuid4())
    request_id = _current_request_id.get() or "internal"
    record = SpanRecord(
        trace_id=trace_id,
        request_id=request_id,
        name=name,
        attributes=redact_payload(attributes),
        started_at=time.time(),
    )
    try:
        yield record
    except Exception as exc:
        record.error = type(exc).__name__
        raise
    finally:
        record.ended_at = time.time()
        _trace_store.add_span(record)
        _langfuse_client.emit_span(record)


def token_count(text: str) -> int:
    return len([token for token in text.split() if token])


def estimate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1k: float,
    output_cost_per_1k: float,
) -> float:
    return (input_tokens / 1000.0 * input_cost_per_1k) + (
        output_tokens / 1000.0 * output_cost_per_1k
    )
