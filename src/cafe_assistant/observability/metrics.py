"""In-process metrics registry used by API, agent, retrieval, and eval code.

This module intentionally keeps metrics dependency-light for local development
and deterministic tests. Runtime code records counters, latency samples, and LLM
cost estimates here; API endpoints can expose a structured snapshot without
logging raw prompts, health facts, phone numbers, or secrets.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class MetricsRegistry:
    """Mutable in-memory store for counters, latency samples, and cost totals.

    The registry is process-local and intentionally simple. It gives tests and
    local deployments a consistent metrics surface, while production can bridge
    the snapshot into a real telemetry backend later.
    """

    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latencies_ms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    cost_usd: float = 0.0

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        """Increment a named counter with deterministic label serialization.

        Args:
            name (str):
                Metric name, such as `requests_total` or
                `retrieval_qdrant_failures_total`.
            value (int):
                Amount to add to the counter.
            **labels (str):
                String labels that distinguish dimensions like route, model, or
                retrieval source kind.

        Returns:
            None:
                The counter is updated in place.
        """
        self.counters[_metric_key(name, labels)] += value

    def observe_latency(self, name: str, duration_ms: float, **labels: str) -> None:
        """Record one latency sample for percentile reporting.

        Args:
            name (str):
                Latency metric name, for example `request_latency_ms`.
            duration_ms (float):
                Observed duration in milliseconds.
            **labels (str):
                String dimensions attached to the latency series.

        Returns:
            None:
                The latency sample is appended to the in-memory series.
        """
        self.latencies_ms[_metric_key(name, labels)].append(duration_ms)

    def add_cost(self, amount_usd: float, **labels: str) -> None:
        """Add an estimated LLM cost and count the corresponding call.

        Args:
            amount_usd (float):
                Estimated cost in US dollars for a model invocation.
            **labels (str):
                String labels such as model name and prompt version.

        Returns:
            None:
                Cost and call counters are updated in place.
        """
        self.cost_usd += amount_usd
        self.counters[_metric_key("llm_calls_total", labels)] += 1

    def snapshot(self) -> dict[str, Any]:
        """Build a structured metrics snapshot grouped by operational concern.

        Args:
            None.

        Returns:
            dict[str, Any]:
                Dictionary containing reliability counters, quality counters,
                latency percentiles, and cost totals. Retrieval fallback counters
                are included under reliability so Qdrant degradation is visible.
        """
        return {
            "reliability": {
                key: value
                for key, value in sorted(self.counters.items())
                if key.startswith(("requests_", "errors_", "rate_limits_", "retrieval_"))
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
        """Clear all metric values from this process-local registry.

        Args:
            None.

        Returns:
            None:
                Counters, latency samples, and accumulated cost are removed.
        """
        self.counters.clear()
        self.latencies_ms.clear()
        self.cost_usd = 0.0


_registry = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    """Return the process-wide metrics registry instance.

    Args:
        None.

    Returns:
        MetricsRegistry:
            Shared registry used by request handlers, retrieval, agent code, and
            tests within the current Python process.
    """
    return _registry


def record_request_result(*, route: str, ok: bool, duration_ms: float) -> None:
    """Record request reliability and latency for one completed route call.

    Args:
        route (str):
            Logical API route or operation name.
        ok (bool):
            Whether the request completed successfully.
        duration_ms (float):
            Total request duration in milliseconds.

    Returns:
        None:
            Request counters and latency samples are updated in the registry.
    """
    registry = get_metrics_registry()
    registry.increment("requests_total", route=route)
    if not ok:
        registry.increment("errors_total", route=route)
    registry.observe_latency("request_latency_ms", duration_ms, route=route)


def record_quality_event(name: str, **labels: str) -> None:
    """Record a quality, safety, or retrieval degradation event counter.

    Args:
        name (str):
            Counter name for the event being recorded.
        **labels (str):
            String dimensions such as reason, stage, source kind, or fallback.

    Returns:
        None:
            The named event counter is incremented by one.
    """
    get_metrics_registry().increment(name, **labels)


def record_llm_cost(amount_usd: float, *, model: str, prompt_version: str) -> None:
    """Record estimated LLM spend for one model call.

    Args:
        amount_usd (float):
            Estimated cost in US dollars.
        model (str):
            Model identifier used for the call.
        prompt_version (str):
            Prompt template version used for the call.

    Returns:
        None:
            Cost and model-call counters are updated in the registry.
    """
    get_metrics_registry().add_cost(amount_usd, model=model, prompt_version=prompt_version)


class RequestTimer:
    """Small helper for measuring request duration with one finish call."""

    def __init__(self, route: str) -> None:
        """Start timing a route or operation.

        Args:
            route (str):
                Logical route or operation name reported with the metrics sample.

        Returns:
            None:
                The start time is captured for later latency calculation.
        """
        self.route = route
        self.started_at = perf_counter()

    def finish(self, *, ok: bool) -> None:
        """Stop timing and record the request result.

        Args:
            ok (bool):
                Whether the measured request or operation succeeded.

        Returns:
            None:
                Reliability and latency metrics are recorded in the global registry.
        """
        record_request_result(
            route=self.route,
            ok=ok,
            duration_ms=(perf_counter() - self.started_at) * 1000.0,
        )


def _metric_key(name: str, labels: dict[str, str]) -> str:
    """Serialize a metric name and labels into a stable dictionary key.

    Args:
        name (str):
            Base metric name.
        labels (dict[str, str]):
            Label names and values attached to the metric.

    Returns:
        str:
            Bare metric name when no labels are provided, otherwise a stable
            `name{key=value,...}` representation sorted by label key.
    """
    if not labels:
        return name
    serialized = ",".join(f"{key}={value}" for key, value in sorted(labels.items()))
    return f"{name}{{{serialized}}}"


def _percentiles(values: list[float]) -> dict[str, float]:
    """Compute the latency summary returned by the metrics endpoint.

    Args:
        values (list[float]):
            Recorded latency samples in milliseconds.

    Returns:
        dict[str, float]:
            Count and p50/p95/p99 values rounded for compact reporting. Empty
            input returns zero values for all percentile fields.
    """
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
    """Select a percentile value from an already sorted sample list.

    Args:
        sorted_values (list[float]):
            Latency samples sorted in ascending order.
        percentile (float):
            Desired percentile as a decimal between 0.0 and 1.0.

    Returns:
        float:
            Sample value nearest to the requested percentile using rounded index
            selection bounded to the available sample range.
    """
    index = max(0, min(len(sorted_values) - 1, round((len(sorted_values) - 1) * percentile)))
    return sorted_values[index]