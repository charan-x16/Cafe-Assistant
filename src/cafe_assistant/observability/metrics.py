"""Process metrics with JSON and OpenMetrics export surfaces.

The application records operational counters, bounded latency samples, and LLM
cost estimates through this module. The in-process registry is intentionally
small and deterministic for tests, while the OpenMetrics renderer gives
production deployments a scrape-friendly format that can be bridged to Prometheus
or another telemetry backend. No raw prompts, health data, phone numbers, or
secrets are accepted as metric labels.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter
from typing import Any

from cafe_assistant.config import settings


@dataclass(slots=True)
class MetricsRegistry:
    """Mutable in-memory store for counters, latency samples, and cost totals.

    The registry is process-local by design, but it keeps memory bounded and can
    render OpenMetrics text for production scraping. Multi-worker deployments
    should scrape each worker or forward these metrics to an external collector.
    """

    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latencies_ms: dict[str, deque[float]] = field(default_factory=dict)
    cost_usd: float = 0.0
    cost_usd_by_key: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _lock: RLock = field(default_factory=RLock, repr=False)

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        """Increment a named counter with deterministic label serialization.

        Args:
            name (str): Metric name, such as `requests_total`.
            value (int): Amount to add to the counter.
            **labels (str): String dimensions such as route, model, or fallback.

        Returns:
            None: The counter is updated in place.
        """
        with self._lock:
            self.counters[_metric_key(name, labels)] += value

    def observe_latency(self, name: str, duration_ms: float, **labels: str) -> None:
        """Record one latency sample in a bounded per-series buffer.

        Args:
            name (str): Latency metric name, for example `request_latency_ms`.
            duration_ms (float): Observed duration in milliseconds.
            **labels (str): String dimensions attached to the latency series.

        Returns:
            None: The sample is appended and older samples are evicted when the
                configured series limit is reached.
        """
        key = _metric_key(name, labels)
        limit = max(1, settings.metrics_latency_sample_limit)
        with self._lock:
            series = self.latencies_ms.get(key)
            if series is None or series.maxlen != limit:
                series = deque(series or [], maxlen=limit)
                self.latencies_ms[key] = series
            series.append(duration_ms)

    def add_cost(self, amount_usd: float, **labels: str) -> None:
        """Add an estimated LLM cost and count the corresponding call.

        Args:
            amount_usd (float): Estimated cost in US dollars for a model invocation.
            **labels (str): String labels such as model name and prompt version.

        Returns:
            None: Cost totals and call counters are updated in place.
        """
        key = _metric_key("llm_estimated_cost_usd_total", labels)
        call_key = _metric_key("llm_calls_total", labels)
        with self._lock:
            self.cost_usd += amount_usd
            self.cost_usd_by_key[key] += amount_usd
            self.counters[call_key] += 1

    def snapshot(self) -> dict[str, Any]:
        """Build a structured metrics snapshot grouped by operational concern.

        Args:
            None.

        Returns:
            dict[str, Any]: Reliability counters, quality counters, latency
                percentiles, and cost totals. Provider fallback/failure counters
                and observability subsystem failures are included so degradation
                is visible from the JSON endpoint.
        """
        with self._lock:
            counters = dict(self.counters)
            latencies = {key: list(values) for key, values in self.latencies_ms.items()}
            cost_total = self.cost_usd
            cost_by_key = dict(self.cost_usd_by_key)
        return {
            "reliability": {
                key: value
                for key, value in sorted(counters.items())
                if key.startswith(
                    (
                        "requests_",
                        "errors_",
                        "rate_limits_",
                        "retrieval_",
                        "llm_provider_",
                        "memory_unavailable_",
                        "recommender_fallback_",
                        "observability_",
                    )
                )
            },
            "quality": {
                key: value
                for key, value in sorted(counters.items())
                if key.startswith(("empty_safe_sets_", "medical_refusals_", "eval_"))
            },
            "latency": {key: _percentiles(values) for key, values in sorted(latencies.items())},
            "cost": {
                "estimated_usd": round(cost_total, 8),
                "estimated_usd_by_model": {
                    key: round(value, 8) for key, value in sorted(cost_by_key.items())
                },
                "llm_calls": {
                    key: value
                    for key, value in sorted(counters.items())
                    if key.startswith("llm_calls_total")
                },
            },
        }

    def to_openmetrics(self) -> str:
        """Render the current registry in OpenMetrics-compatible text format.

        Args:
            None.

        Returns:
            str: Text payload ending with `# EOF` for scrape clients. Counter
                labels are preserved, latency percentiles are exported as gauges,
                and cost is exported both globally and by model/prompt labels.
        """
        with self._lock:
            counters = dict(self.counters)
            latencies = {key: list(values) for key, values in self.latencies_ms.items()}
            cost_total = self.cost_usd
            cost_by_key = dict(self.cost_usd_by_key)
        lines: list[str] = []
        emitted_types: set[str] = set()
        for key, value in sorted(counters.items()):
            name, labels = _split_metric_key(key)
            _emit_type(lines, emitted_types, name, "counter")
            lines.append(f"{_sanitize_metric_name(name)}{_format_labels(labels)} {value}")
        _emit_type(lines, emitted_types, "llm_estimated_cost_usd_total", "counter")
        lines.append(f"llm_estimated_cost_usd_total {round(cost_total, 8)}")
        for key, value in sorted(cost_by_key.items()):
            name, labels = _split_metric_key(key)
            _emit_type(lines, emitted_types, name, "counter")
            lines.append(f"{_sanitize_metric_name(name)}{_format_labels(labels)} {round(value, 8)}")
        for key, values in sorted(latencies.items()):
            name, labels = _split_metric_key(key)
            summary = _percentiles(values)
            for percentile in ("p50", "p95", "p99"):
                metric_name = f"{name}_{percentile}"
                _emit_type(lines, emitted_types, metric_name, "gauge")
                lines.append(
                    f"{_sanitize_metric_name(metric_name)}{_format_labels(labels)} "
                    f"{summary[percentile]}"
                )
            count_name = f"{name}_count"
            _emit_type(lines, emitted_types, count_name, "gauge")
            lines.append(
                f"{_sanitize_metric_name(count_name)}{_format_labels(labels)} {summary['count']}"
            )
        lines.append("# EOF")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Clear all metric values from this process-local registry.

        Args:
            None.

        Returns:
            None: Counters, latency samples, and accumulated costs are removed.
        """
        with self._lock:
            self.counters.clear()
            self.latencies_ms.clear()
            self.cost_usd = 0.0
            self.cost_usd_by_key.clear()


_registry = MetricsRegistry()


def get_metrics_registry() -> MetricsRegistry:
    """Return the process-wide metrics registry instance.

    Args:
        None: The registry is a module-level singleton.

    Returns:
        MetricsRegistry: Shared registry used by request handlers, retrieval,
            agent code, evals, and tests within the current Python process.
    """
    return _registry


def record_request_result(*, route: str, ok: bool, duration_ms: float) -> None:
    """Record request reliability and latency for one completed route call.

    Args:
        route (str): Logical API route or operation name.
        ok (bool): Whether the request completed successfully.
        duration_ms (float): Total request duration in milliseconds.

    Returns:
        None: Request counters and latency samples are updated in the registry.
    """
    registry = get_metrics_registry()
    registry.increment("requests_total", route=route)
    if not ok:
        registry.increment("errors_total", route=route)
    registry.observe_latency("request_latency_ms", duration_ms, route=route)


def record_quality_event(name: str, **labels: str) -> None:
    """Record a quality, safety, degradation, or observability event counter.

    Args:
        name (str): Counter name for the event being recorded.
        **labels (str): String dimensions such as reason, stage, source kind, or fallback.

    Returns:
        None: The named event counter is incremented by one.
    """
    get_metrics_registry().increment(name, **labels)


def record_llm_cost(amount_usd: float, *, model: str, prompt_version: str) -> None:
    """Record estimated LLM spend for one model call.

    Args:
        amount_usd (float): Estimated cost in US dollars.
        model (str): Model identifier used for the call.
        prompt_version (str): Prompt template version used for the call.

    Returns:
        None: Cost totals and model-call counters are updated in the registry.
    """
    get_metrics_registry().add_cost(amount_usd, model=model, prompt_version=prompt_version)


class RequestTimer:
    """Small helper for measuring request duration with one finish call."""

    def __init__(self, route: str) -> None:
        """Start timing a route or operation.

        Args:
            route (str): Logical route or operation name reported with metrics.

        Returns:
            None: The start time is captured for later latency calculation.
        """
        self.route = route
        self.started_at = perf_counter()

    def finish(self, *, ok: bool) -> None:
        """Stop timing and record the request result.

        Args:
            ok (bool): Whether the measured request or operation succeeded.

        Returns:
            None: Reliability and latency metrics are recorded in the registry.
        """
        record_request_result(
            route=self.route,
            ok=ok,
            duration_ms=(perf_counter() - self.started_at) * 1000.0,
        )


def _metric_key(name: str, labels: dict[str, str]) -> str:
    """Serialize a metric name and labels into a stable dictionary key.

    Args:
        name (str): Base metric name.
        labels (dict[str, str]): Label names and values attached to the metric.

    Returns:
        str: Bare metric name when no labels are provided, otherwise a stable
            `name{key=value,...}` representation sorted by label key.
    """
    if not labels:
        return name
    serialized = ",".join(f"{key}={value}" for key, value in sorted(labels.items()))
    return f"{name}{{{serialized}}}"


def _split_metric_key(key: str) -> tuple[str, dict[str, str]]:
    """Split an internal metric key back into name and labels.

    Args:
        key (str): Serialized metric key produced by `_metric_key`.

    Returns:
        tuple[str, dict[str, str]]: Metric name and parsed label dictionary.
    """
    if "{" not in key or not key.endswith("}"):
        return key, {}
    name, label_blob = key[:-1].split("{", 1)
    labels: dict[str, str] = {}
    for pair in label_blob.split(","):
        if "=" not in pair:
            continue
        label_name, label_value = pair.split("=", 1)
        labels[label_name] = label_value
    return name, labels


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    """Compute the latency summary returned by the metrics endpoint.

    Args:
        values (Iterable[float]): Recorded latency samples in milliseconds.

    Returns:
        dict[str, float]: Count and p50/p95/p99 values rounded for compact reporting.
    """
    sorted_values = sorted(values)
    if not sorted_values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": len(sorted_values),
        "p50": round(_percentile(sorted_values, 0.50), 3),
        "p95": round(_percentile(sorted_values, 0.95), 3),
        "p99": round(_percentile(sorted_values, 0.99), 3),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Select a percentile value from an already sorted sample list.

    Args:
        sorted_values (list[float]): Latency samples sorted in ascending order.
        percentile (float): Desired percentile as a decimal between 0.0 and 1.0.

    Returns:
        float: Sample value nearest to the requested percentile.
    """
    index = max(0, min(len(sorted_values) - 1, round((len(sorted_values) - 1) * percentile)))
    return sorted_values[index]


def _emit_type(lines: list[str], emitted_types: set[str], name: str, metric_type: str) -> None:
    """Emit an OpenMetrics TYPE line once for a metric name.

    Args:
        lines (list[str]): Output lines being built.
        emitted_types (set[str]): Metric names that already emitted a type line.
        name (str): Metric name to describe.
        metric_type (str): OpenMetrics type such as `counter` or `gauge`.

    Returns:
        None: The TYPE line is appended only when needed.
    """
    sanitized = _sanitize_metric_name(name)
    if sanitized in emitted_types:
        return
    emitted_types.add(sanitized)
    lines.append(f"# TYPE {sanitized} {metric_type}")


def _format_labels(labels: dict[str, str]) -> str:
    """Format labels for one OpenMetrics sample.

    Args:
        labels (dict[str, str]): Label names and values.

    Returns:
        str: Empty string for no labels, otherwise `{name="value"}` text.
    """
    if not labels:
        return ""
    rendered = ",".join(
        f'{_sanitize_label_name(name)}="{_escape_label_value(value)}"'
        for name, value in sorted(labels.items())
    )
    return "{" + rendered + "}"


def _sanitize_metric_name(name: str) -> str:
    """Convert a metric name into a Prometheus-safe identifier.

    Args:
        name (str): Internal metric name.

    Returns:
        str: Name containing only letters, digits, underscores, and colons.
    """
    sanitized = "".join(char if char.isalnum() or char in "_:" else "_" for char in name)
    if not sanitized or sanitized[0].isdigit():
        return f"_{sanitized}"
    return sanitized


def _sanitize_label_name(name: str) -> str:
    """Convert a label name into a Prometheus-safe identifier.

    Args:
        name (str): Internal label name.

    Returns:
        str: Label name containing only letters, digits, and underscores.
    """
    sanitized = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    if not sanitized or sanitized[0].isdigit():
        return f"_{sanitized}"
    return sanitized


def _escape_label_value(value: str) -> str:
    """Escape a label value for OpenMetrics text output.

    Args:
        value (str): Raw label value.

    Returns:
        str: Value with backslashes, quotes, and newlines escaped.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
