"""Incident replay helpers for stored in-process trace records.

Replay is useful for debugging because it reconstructs prompt context, retrieved
item IDs, tool spans, and version metadata. It is also sensitive operational
metadata, so callers may optionally supply a tenant ID; mismatched tenants are
reported as not found to avoid cross-tenant trace discovery.
"""

from __future__ import annotations

from typing import Any

from cafe_assistant.observability.tracing import TraceRecord, get_trace_store


def replay_trace(trace_id: str, *, tenant_id: int | None = None) -> dict[str, Any]:
    """Replay one stored trace, optionally constrained to a tenant.

    Args:
        trace_id (str):
            Trace identifier captured during request processing.
        tenant_id (int | None):
            Tenant that must own the trace. `None` is reserved for trusted local
            callers such as tests and command-line debugging helpers.

    Returns:
        dict[str, Any]:
            Structured replay payload with redacted prompt context, tool spans,
            retrieved item IDs, and version metadata.
    """
    trace = get_trace_store().get(trace_id)
    if trace is None or (tenant_id is not None and trace.tenant_id != tenant_id):
        raise TraceNotFoundError(trace_id)
    return replay_trace_record(trace)


def replay_trace_record(trace: TraceRecord) -> dict[str, Any]:
    """Convert a trace record into a replay-safe response payload.

    Args:
        trace (TraceRecord):
            In-memory trace record whose span attributes were redacted at capture time.

    Returns:
        dict[str, Any]:
            JSON-compatible payload for incident debugging and tests.
    """
    return {
        "trace_id": trace.trace_id,
        "request_id": trace.request_id,
        "tenant_id": trace.tenant_id,
        "version_registry": trace.version_registry,
        "prompt_context": [
            span.attributes
            for span in trace.spans
            if span.name in {"llm.compose", "llm.classify"}
        ],
        "retrieved_items": [
            span.attributes.get("retrieved_item_ids")
            for span in trace.spans
            if span.attributes.get("retrieved_item_ids") is not None
        ],
        "tools": [span.attributes for span in trace.spans if span.name.startswith("tool.")],
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
        "spans": [
            {
                "name": span.name,
                "duration_ms": span.duration_ms,
                "attributes": span.attributes,
                "error": span.error,
            }
            for span in trace.spans
        ],
    }


class TraceNotFoundError(KeyError):
    """Raised when a trace is missing or not owned by the requested tenant."""
