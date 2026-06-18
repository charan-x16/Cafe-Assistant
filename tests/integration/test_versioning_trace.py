from __future__ import annotations

from cafe_assistant.observability.replay import replay_trace
from cafe_assistant.observability.tracing import start_trace


def test_trace_records_release_version_registry() -> None:
    start_trace(tenant_id=1, request_id="req-version", trace_id="trace-version")

    payload = replay_trace("trace-version")
    registry = payload["version_registry"]

    assert registry["prompts"]["composer"] == "composer_v1"
    assert registry["tools"]["search_menu"] == "search_menu_v1"
    assert registry["retrievers"]["hybrid"] == "rrf_v1"
    assert registry["policy_rules"]["dietary_safety"] == "unknown_unsafe_v1"
    assert registry["memory_write_rules"]["health_data_consent_gate"] == "dietary_health_consent_v1"
    assert registry["orchestrator_graph"]["name"] == "custom_fsm"
