from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class MetricsRegistry:
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latencies_ms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    cost_usd: float = 0.0

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        self.counters[_metric_key(name, labels)] += value

    def observe_latency(self, name: str, duration_ms: float, **labels: str) -> None:
        self.latencies_ms[_metric_key(name, labels)].append(duration_ms)

    def add_cost(self, amount_usd: float, **labels: str) -> None:
        self.cost_usd += amount_usd
        self.counters[_metric_key("llm_calls_total", labels)] += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "reliability": {
                key: value
                for key, value in sorted(self.counters.items())
                if key.startswith(("requests_", "errors_", "rate_limits_"))
            },
            "quality": {
                key: value
                for key, value in sorted(self.counters.items())
                if key.startswith(("empty_safe_sets_", "medical_refusals_", "eval_"))
            },
            "latency": {
                key: _percentiles(values)
                for key, values in sorted(self.latencies_ms.items())
            },
            "cost": {
                "estimated_usd": round(self.cost_usd, 8),
                "llm_calls": {
                    key: value
                    for key, value in sorted(self.counters.items())
                    if key.startswith("llm_calls_total")
                },
            },
        }

    def reset(self) -> None:
        self.counters.clear()
        self.latencies_ms.clear()
        self.cost_usd = 0.0


_registry = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    return _registry


def record_request_result(*, route: str, ok: bool, duration_ms: float) -> None:
    registry = get_metrics_registry()
    registry.increment("requests_total", route=route)
    if not ok:
        registry.increment("errors_total", route=route)
    registry.observe_latency("request_latency_ms", duration_ms, route=route)


def record_quality_event(name: str, **labels: str) -> None:
    get_metrics_registry().increment(name, **labels)


def record_llm_cost(amount_usd: float, *, model: str, prompt_version: str) -> None:
    get_metrics_registry().add_cost(amount_usd, model=model, prompt_version=prompt_version)


class RequestTimer:
    def __init__(self, route: str) -> None:
        self.route = route
        self.started_at = perf_counter()

    def finish(self, *, ok: bool) -> None:
        record_request_result(
            route=self.route,
            ok=ok,
            duration_ms=(perf_counter() - self.started_at) * 1000.0,
        )


def _metric_key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    serialized = ",".join(f"{key}={value}" for key, value in sorted(labels.items()))
    return f"{name}{{{serialized}}}"


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "p50": round(_percentile(sorted_values, 0.50), 3),
        "p95": round(_percentile(sorted_values, 0.95), 3),
        "p99": round(_percentile(sorted_values, 0.99), 3),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    index = max(0, min(len(sorted_values) - 1, round((len(sorted_values) - 1) * percentile)))
    return sorted_values[index]
