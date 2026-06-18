from __future__ import annotations

from typing import Any

from cafe_assistant.observability.tracing import TraceRecord, get_trace_store


def replay_trace(trace_id: str) -> dict[str, Any]:
    trace = get_trace_store().get(trace_id)
    if trace is None:
        raise TraceNotFoundError(trace_id)
    return replay_trace_record(trace)


def replay_trace_record(trace: TraceRecord) -> dict[str, Any]:
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
        "tools": [
            span.attributes
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
            )
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
    pass
