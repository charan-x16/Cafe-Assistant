"""Tests for production observability guarantees.

These tests cover the behavior needed during incidents: traces must survive beyond
the in-memory cache, replay must remain redacted even when span attributes are
mutated after creation, duplicate trace starts must not erase spans, and metrics
must expose provider degradation plus bounded latency data.
"""

from __future__ import annotations

import json
from pathlib import Path

from cafe_assistant.config import settings
from cafe_assistant.observability.metrics import (
    get_metrics_registry,
    record_llm_cost,
    record_quality_event,
    record_request_result,
)
from cafe_assistant.observability.replay import replay_trace
from cafe_assistant.observability.tracing import finish_trace, get_trace_store, span, start_trace


def test_trace_replay_loads_from_durable_store_and_redacts_late_attributes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Assert durable replay works after memory is cleared and remains redacted.

    Args:
        tmp_path (Path): Temporary directory used as the durable trace spool.
        monkeypatch: Pytest fixture used to point settings at the temporary spool.

    Returns:
        None: Failed expectations raise pytest assertion errors.
    """
    monkeypatch.setattr(
        settings,
        "observability_trace_store_path",
        str(tmp_path / "traces.jsonl"),
    )
    trace_store = get_trace_store()
    trace_store.reset(clear_durable=True)

    trace_id = start_trace(tenant_id=7, request_id="req-durable", trace_id="trace-durable")
    with span(
        "llm.compose",
        prompt_messages=[
            {
                "role": "user",
                "content": "I am allergic to peanuts and my phone is +1 555 123 4567.",
            }
        ],
    ) as record:
        record.attributes.update(
            {
                "late_secret": "api_key=sk-test-secret",
                "phone": "+1 555 999 0000",
            }
        )
    finish_trace(trace_id)

    trace_store.reset(clear_durable=False)
    payload = replay_trace("trace-durable", tenant_id=7)
    serialized = json.dumps(payload)

    assert payload["tenant_id"] == 7
    assert payload["duration_ms"] is not None
    assert payload["prompt_context"]
    assert "sk-test-secret" not in serialized
    assert "+1 555" not in serialized
    assert "peanuts" not in serialized.lower()
    assert "[REDACTED]" in serialized or "[REDACTED_HEALTH]" in serialized


def test_start_trace_is_idempotent_for_existing_trace(tmp_path: Path, monkeypatch) -> None:
    """Assert duplicate trace starts do not erase existing spans.

    Args:
        tmp_path (Path): Temporary directory used as the durable trace spool.
        monkeypatch: Pytest fixture used to point settings at the temporary spool.

    Returns:
        None: Failed expectations raise pytest assertion errors.
    """
    monkeypatch.setattr(
        settings,
        "observability_trace_store_path",
        str(tmp_path / "idempotent-traces.jsonl"),
    )
    trace_store = get_trace_store()
    trace_store.reset(clear_durable=True)

    start_trace(tenant_id=3, request_id="req-idem", trace_id="trace-idem")
    with span("router.classify") as first_span:
        first_span.attributes["route"] = "menu_qa"
    start_trace(tenant_id=3, request_id="req-idem", trace_id="trace-idem")
    with span("agent.filtering", retrieved_item_ids=[1, 2]):
        pass
    finish_trace("trace-idem")

    payload = replay_trace("trace-idem", tenant_id=3)
    assert [span_payload["name"] for span_payload in payload["spans"]] == [
        "router.classify",
        "agent.filtering",
    ]
    assert payload["route"] == "menu_qa"


def test_metrics_snapshot_and_openmetrics_include_degradation_and_bounded_latency(
    monkeypatch,
) -> None:
    """Assert metrics expose provider failures and keep latency samples bounded.

    Args:
        monkeypatch: Pytest fixture used to reduce the latency sample limit.

    Returns:
        None: Failed expectations raise pytest assertion errors.
    """
    monkeypatch.setattr(settings, "metrics_latency_sample_limit", 2)
    registry = get_metrics_registry()
    registry.reset()

    record_request_result(route="/chat", ok=True, duration_ms=100.0)
    record_request_result(route="/chat", ok=True, duration_ms=200.0)
    record_request_result(route="/chat", ok=True, duration_ms=300.0)
    record_quality_event(
        "llm_provider_failures_total",
        provider="openai",
        model="gpt-4o-mini",
        retryable="true",
    )
    record_quality_event("observability_failures_total", reason="langfuse_export")
    record_llm_cost(0.012345, model="gpt-4o-mini", prompt_version="composer_v1")

    snapshot = registry.snapshot()
    openmetrics = registry.to_openmetrics()

    assert snapshot["latency"]["request_latency_ms{route=/chat}"]["count"] == 2
    assert (
        snapshot["reliability"][
            "llm_provider_failures_total{model=gpt-4o-mini,provider=openai,retryable=true}"
        ]
        == 1
    )
    assert "observability_failures_total{reason=langfuse_export}" in snapshot["reliability"]
    assert "llm_provider_failures_total" in openmetrics
    assert "request_latency_ms_p95" in openmetrics
    assert "llm_estimated_cost_usd_total" in openmetrics
    assert openmetrics.endswith("# EOF\n")

    registry.reset()
