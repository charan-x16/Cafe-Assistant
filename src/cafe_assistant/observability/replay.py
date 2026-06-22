"""Incident replay helpers for redacted request traces.

Replay is used after an incident to reconstruct which route ran, which tools were
called, which menu item IDs were retrieved, which prompt/tool/model versions were
active, and which redacted model context was sent. The trace store checks the
in-memory cache first and then the durable JSONL spool, so replay can work from a
separate process as long as it can read the configured trace store path.
"""

from __future__ import annotations

from typing import Any

from cafe_assistant.observability.tracing import SpanRecord, TraceRecord, get_trace_store
from cafe_assistant.security.redaction import redact_payload


def replay_trace(trace_id: str, *, tenant_id: int | None = None) -> dict[str, Any]:
    """Replay one stored trace, optionally constrained to a tenant.

    Args:
        trace_id (str): Trace identifier captured during request processing.
        tenant_id (int | None): Tenant that must own the trace. `None` is reserved
            for trusted local callers such as tests and command-line debugging helpers.

    Returns:
        dict[str, Any]: Structured replay payload with redacted prompt context,
            tool spans, retrieved item IDs, routing metadata, and version metadata.
    """
    trace = get_trace_store().get(trace_id)
    if trace is None or (tenant_id is not None and trace.tenant_id != tenant_id):
        raise TraceNotFoundError(trace_id)
    return replay_trace_record(trace)


def replay_trace_record(trace: TraceRecord) -> dict[str, Any]:
    """Convert a trace record into a replay-safe response payload.

    Args:
        trace (TraceRecord): Trace record from memory or durable storage.

    Returns:
        dict[str, Any]: JSON-compatible payload for incident debugging and tests.
    """
    prompt_spans = [span for span in trace.spans if span.name in {"llm.compose", "llm.classify"}]
    router_span = _last_span_named(trace.spans, "router.classify")
    return {
        "trace_id": trace.trace_id,
        "request_id": trace.request_id,
        "tenant_id": trace.tenant_id,
        "started_at": trace.started_at,
        "ended_at": trace.ended_at,
        "duration_ms": trace.duration_ms,
        "route": router_span.attributes.get("route") if router_span is not None else None,
        "route_confidence": (
            router_span.attributes.get("confidence") if router_span is not None else None
        ),
        "version_registry": trace.version_registry,
        "prompt_context": [redact_payload(span.attributes) for span in prompt_spans],
        "retrieved_items": [
            span.attributes.get("retrieved_item_ids")
            for span in trace.spans
            if span.attributes.get("retrieved_item_ids") is not None
        ],
        "tools": [
            _span_to_replay_payload(span)
            for span in trace.spans
            if span.name.startswith("tool.")
        ],
        "versions": {
            "registry": trace.version_registry,
            "prompt_versions": sorted(
                {
                    str(span.attributes.get("prompt_version"))
                    for span in trace.spans
                    if span.attributes.get("prompt_version") is not None
                }
            ),
        },
        "spans": [_span_to_replay_payload(span) for span in trace.spans],
    }


def _span_to_replay_payload(span: SpanRecord) -> dict[str, Any]:
    """Convert a span to the redacted shape returned by replay.

    Args:
        span (SpanRecord): Span captured during request handling.

    Returns:
        dict[str, Any]: Span metadata, duration, parent/child identifiers, and
            redacted attributes.
    """
    return {
        "span_id": span.span_id,
        "parent_span_id": span.parent_span_id,
        "name": span.name,
        "duration_ms": span.duration_ms,
        "attributes": redact_payload(span.attributes),
        "error": span.error,
    }


def _last_span_named(spans: list[SpanRecord], name: str) -> SpanRecord | None:
    """Return the last span with a given name.

    Args:
        spans (list[SpanRecord]): Ordered spans in a trace.
        name (str): Span name to search for.

    Returns:
        SpanRecord | None: Most recent matching span, or None when absent.
    """
    for span in reversed(spans):
        if span.name == name:
            return span
    return None


class TraceNotFoundError(KeyError):
    """Raised when a trace is missing or not owned by the requested tenant."""
