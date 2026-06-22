"""Request tracing, durable trace snapshots, and optional Langfuse export.

The cafe assistant uses this module for operational debugging and incident
replay. Traces are intentionally lightweight: runtime code records redacted span
attributes into a bounded in-memory cache for fast API reads, while the same
trace snapshots are appended to a JSONL spool when traces start and finish so a
later process or command-line helper can replay a completed incident after the
original worker has finished. Langfuse is
an optional exporter; failures to export are recorded as safe metrics and never
block the customer request path.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from cafe_assistant.config import settings
from cafe_assistant.security.redaction import redact_payload
from cafe_assistant.versioning import get_version_registry

logger = logging.getLogger(__name__)

_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cafe_assistant_trace_id",
    default=None,
)
_current_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cafe_assistant_request_id",
    default=None,
)
_current_tenant_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "cafe_assistant_tenant_id",
    default=None,
)
_current_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cafe_assistant_span_id",
    default=None,
)


@dataclass(slots=True)
class SpanRecord:
    """One timed operation inside a request trace.

    Attributes:
        trace_id (str): Request-level trace identifier shared by all spans.
        request_id (str): Public request identifier propagated through API headers.
        name (str): Logical span name such as `router.classify` or `llm.compose`.
        attributes (dict[str, Any]): Redacted operational metadata for replay.
        started_at (float): Unix timestamp captured when the span started.
        span_id (str): Unique span identifier used to reconstruct nesting.
        parent_span_id (str | None): Parent span identifier, if the span is nested.
        ended_at (float | None): Unix timestamp captured when the span ended.
        error (str | None): Exception type name when the span failed.
    """

    trace_id: str
    request_id: str
    name: str
    attributes: dict[str, Any]
    started_at: float
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: str | None = None
    ended_at: float | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        """Return the span duration in milliseconds when the span has ended.

        Args:
            None: Duration is derived from `started_at` and `ended_at`.

        Returns:
            float | None: Milliseconds between start and end, or None for an
                unfinished span.
        """
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000.0

    def to_payload(self) -> dict[str, Any]:
        """Serialize the span into a JSON-compatible replay payload.

        Args:
            None: The span instance supplies all serialized values.

        Returns:
            dict[str, Any]: Redacted span fields safe to append to the durable
                trace spool or return through the replay endpoint.
        """
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "attributes": redact_payload(self.attributes),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SpanRecord:
        """Hydrate a span record from a durable JSONL payload.

        Args:
            payload (Mapping[str, Any]): JSON object previously produced by
                `to_payload`.

        Returns:
            SpanRecord: In-memory span with redaction applied again on load.
        """
        return cls(
            trace_id=str(payload.get("trace_id", "")),
            request_id=str(payload.get("request_id", "internal")),
            span_id=str(payload.get("span_id") or uuid.uuid4()),
            parent_span_id=(
                str(payload["parent_span_id"])
                if payload.get("parent_span_id") is not None
                else None
            ),
            name=str(payload.get("name", "unknown")),
            attributes=redact_payload(_mapping_or_empty(payload.get("attributes"))),
            started_at=_float_value(payload.get("started_at"), default=time.time()),
            ended_at=_optional_float(payload.get("ended_at")),
            error=str(payload["error"]) if payload.get("error") is not None else None,
        )


@dataclass(slots=True)
class TraceRecord:
    """Request-level trace containing version metadata and timed spans.

    Attributes:
        trace_id (str): Stable trace identifier propagated through the request.
        request_id (str): Public request identifier returned to clients.
        tenant_id (int): Tenant that owns the trace for replay authorization.
        version_registry (dict[str, object]): Prompt, tool, model, retriever,
            policy, memory, and graph versions active for the request.
        started_at (float): Unix timestamp when the trace was first opened.
        ended_at (float | None): Unix timestamp when the request completed.
        spans (list[SpanRecord]): Ordered span records captured for the request.
    """

    trace_id: str
    request_id: str
    tenant_id: int
    version_registry: dict[str, object]
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    spans: list[SpanRecord] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        """Return total trace duration in milliseconds when the trace is complete.

        Args:
            None: Duration is derived from trace start and finish timestamps.

        Returns:
            float | None: Milliseconds between trace start and finish, or None
                when `finish_trace` has not been called.
        """
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000.0

    def to_payload(self) -> dict[str, Any]:
        """Serialize the complete trace into a JSON-compatible payload.

        Args:
            None: The trace instance supplies all serialized values.

        Returns:
            dict[str, Any]: Redacted trace data suitable for durable storage and
                incident replay.
        """
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "version_registry": self.version_registry,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "spans": [span.to_payload() for span in self.spans],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TraceRecord:
        """Hydrate a trace record from durable storage.

        Args:
            payload (Mapping[str, Any]): JSON object previously produced by
                `to_payload`.

        Returns:
            TraceRecord: Reconstructed trace with all span attributes redacted.
        """
        spans = [
            SpanRecord.from_payload(span_payload)
            for span_payload in payload.get("spans", [])
            if isinstance(span_payload, Mapping)
        ]
        version_registry = payload.get("version_registry")
        if not isinstance(version_registry, dict):
            version_registry = {}
        return cls(
            trace_id=str(payload.get("trace_id", "")),
            request_id=str(payload.get("request_id", "internal")),
            tenant_id=int(payload.get("tenant_id") or 0),
            version_registry=version_registry,
            started_at=_float_value(payload.get("started_at"), default=time.time()),
            ended_at=_optional_float(payload.get("ended_at")),
            spans=spans,
        )


class DurableTraceStore:
    """Append-only JSONL trace spool used for cross-process incident replay."""

    def path(self) -> Path | None:
        """Resolve the configured durable trace path.

        Args:
            None: The path is read from `settings.observability_trace_store_path`.

        Returns:
            Path | None: Expanded filesystem path, or None when durable tracing
                is disabled by an empty setting.
        """
        configured = settings.observability_trace_store_path.strip()
        if not configured:
            return None
        return Path(configured).expanduser()

    def persist(self, trace: TraceRecord) -> None:
        """Append the latest trace snapshot to the durable JSONL spool.

        Args:
            trace (TraceRecord): Trace snapshot to serialize and append.

        Returns:
            None: Storage failures are converted into observability metrics and
                redacted warning logs so request handling can continue.
        """
        path = self.path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            event = {
                "written_at": time.time(),
                "trace_id": trace.trace_id,
                "record": trace.to_payload(),
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, separators=(",", ":"), default=str))
                handle.write("\n")
        except OSError as exc:
            _record_observability_failure("trace_store_write", exc)

    def load(self, trace_id: str) -> TraceRecord | None:
        """Load the latest durable snapshot for one trace ID.

        Args:
            trace_id (str): Trace identifier to search for in the JSONL spool.

        Returns:
            TraceRecord | None: Latest matching trace snapshot, or None when no
                durable record exists or durable tracing is disabled.
        """
        path = self.path()
        if path is None or not path.exists():
            return None
        latest: Mapping[str, Any] | None = None
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("trace_id") == trace_id and isinstance(event.get("record"), dict):
                        latest = event["record"]
        except OSError as exc:
            _record_observability_failure("trace_store_read", exc)
            return None
        return TraceRecord.from_payload(latest) if latest is not None else None

    def clear(self) -> None:
        """Delete the durable trace spool for isolated tests or local cleanup.

        Args:
            None: The configured path identifies the file to remove.

        Returns:
            None: Missing files are ignored; deletion failures are surfaced as
                observability failure metrics.
        """
        path = self.path()
        if path is None or not path.exists():
            return
        try:
            path.unlink()
        except OSError as exc:
            _record_observability_failure("trace_store_clear", exc)


class TraceStore:
    """Thread-safe trace cache backed by a finish-time durable spool."""

    def __init__(
        self,
        *,
        max_traces: int = 500,
        durable_store: DurableTraceStore | None = None,
    ) -> None:
        """Create an in-memory trace cache.

        Args:
            max_traces (int): Maximum number of recent traces retained in memory.
            durable_store (DurableTraceStore | None): Optional durable spool. When
                omitted, the default JSONL store reads the configured path.

        Returns:
            None: The cache starts empty and fills as traces are recorded.
        """
        self.max_traces = max(1, max_traces)
        self._durable_store = durable_store or DurableTraceStore()
        self._traces: dict[str, TraceRecord] = {}
        self._order: list[str] = []
        self._lock = RLock()

    def start_trace(self, *, trace_id: str, request_id: str, tenant_id: int) -> TraceRecord:
        """Open a trace or return the existing trace with the same ID.

        Args:
            trace_id (str): Trace identifier from the request context.
            request_id (str): Request identifier from the request context.
            tenant_id (int): Tenant that owns the request and replay permission.

        Returns:
            TraceRecord: Existing or newly created trace. Existing traces are not
                cleared, which makes duplicate starts safe across API and agent boundaries.
        """
        with self._lock:
            existing = self._traces.get(trace_id)
            if existing is not None and existing.ended_at is None:
                return existing
            record = TraceRecord(
                trace_id=trace_id,
                request_id=request_id,
                tenant_id=tenant_id,
                version_registry=get_version_registry().as_trace_attributes(),
            )
            self._remember(record)
            self._durable_store.persist(record)
            return record

    def add_span(self, span: SpanRecord, *, tenant_id: int = 0) -> None:
        """Append a span to its trace without blocking on durable I/O.

        Args:
            span (SpanRecord): Completed span to add to the trace.
            tenant_id (int): Tenant used only when the span created an internal trace.

        Returns:
            None: The in-memory trace cache is updated in place. The durable
                snapshot is written by `finish_trace` so first-token latency is
                not affected by per-span filesystem writes.
        """
        with self._lock:
            trace = self._traces.get(span.trace_id)
            if trace is None:
                trace = TraceRecord(
                    trace_id=span.trace_id,
                    request_id=span.request_id,
                    tenant_id=tenant_id,
                    version_registry=get_version_registry().as_trace_attributes(),
                )
                self._remember(trace)
            trace.spans.append(span)

    def finish_trace(self, trace_id: str) -> TraceRecord | None:
        """Mark a trace complete and persist a final snapshot.

        Args:
            trace_id (str): Trace identifier to finish.

        Returns:
            TraceRecord | None: Finished trace, or None when no trace exists.
        """
        with self._lock:
            trace = self._traces.get(trace_id) or self._durable_store.load(trace_id)
            if trace is None:
                return None
            trace.ended_at = time.time()
            self._remember(trace)
            self._durable_store.persist(trace)
            return trace

    def get(self, trace_id: str) -> TraceRecord | None:
        """Return a trace from memory or durable storage.

        Args:
            trace_id (str): Trace identifier to retrieve.

        Returns:
            TraceRecord | None: Matching trace snapshot, or None when no memory
                or durable record exists.
        """
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is not None:
                return trace
            durable_trace = self._durable_store.load(trace_id)
            if durable_trace is not None:
                self._remember(durable_trace)
            return durable_trace

    def reset(self, *, clear_durable: bool = False) -> None:
        """Clear in-memory traces and optionally delete the durable spool.

        Args:
            clear_durable (bool): Whether to remove the configured JSONL file.

        Returns:
            None: The cache is empty after the call.
        """
        with self._lock:
            self._traces.clear()
            self._order.clear()
            if clear_durable:
                self._durable_store.clear()

    def _remember(self, trace: TraceRecord) -> None:
        """Store a trace in memory while enforcing the bounded cache size.

        Args:
            trace (TraceRecord): Trace snapshot to retain.

        Returns:
            None: Internal cache state is updated in place.
        """
        self._traces[trace.trace_id] = trace
        if trace.trace_id in self._order:
            self._order.remove(trace.trace_id)
        self._order.append(trace.trace_id)
        while len(self._order) > self.max_traces:
            oldest = self._order.pop(0)
            self._traces.pop(oldest, None)


class LangfuseClient:
    """Optional Langfuse adapter that never blocks request handling."""

    def __init__(self) -> None:
        """Create the Langfuse client when it is enabled and importable.

        Args:
            None: Configuration is read from the global settings object.

        Returns:
            None: Initialization failures leave the exporter disabled and record
                a safe operational signal.
        """
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
        except Exception as exc:  # noqa: BLE001 - exporter setup must not fail startup.
            self._client = None
            _record_observability_failure("langfuse_init", exc)

    def emit_span(self, span: SpanRecord) -> None:
        """Export one redacted span to Langfuse when the exporter is enabled.

        Args:
            span (SpanRecord): Completed span whose attributes have already been redacted.

        Returns:
            None: Export failures are counted and logged without raising.
        """
        if self._client is None:
            return
        try:
            trace = self._client.trace(  # type: ignore[attr-defined]
                id=span.trace_id,
                name="cafe_assistant_request",
                metadata={"request_id": span.request_id},
            )
            trace.span(
                id=span.span_id,
                name=span.name,
                metadata=redact_payload(span.attributes),
                start_time=_datetime_from_timestamp(span.started_at),
                end_time=(
                    _datetime_from_timestamp(span.ended_at)
                    if span.ended_at is not None
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must never break chat.
            _record_observability_failure("langfuse_export", exc)


_trace_store = TraceStore(max_traces=settings.trace_store_max_traces)
_langfuse_client = LangfuseClient()


def get_trace_store() -> TraceStore:
    """Return the process-wide trace store.

    Args:
        None: The store is a module-level singleton.

    Returns:
        TraceStore: Shared in-memory cache backed by the configured finish-time durable spool.
    """
    return _trace_store


def start_trace(*, tenant_id: int, request_id: str, trace_id: str | None = None) -> str:
    """Start or attach to a request trace and bind it to the current context.

    Args:
        tenant_id (int): Tenant that owns the request and replay permission.
        request_id (str): Request identifier propagated to clients and logs.
        trace_id (str | None): Optional caller-supplied trace ID. A new UUID is
            generated when omitted.

    Returns:
        str: The trace ID bound to the current context.
    """
    resolved_trace_id = trace_id or str(uuid.uuid4())
    _current_trace_id.set(resolved_trace_id)
    _current_request_id.set(request_id)
    _current_tenant_id.set(tenant_id)
    _trace_store.start_trace(
        trace_id=resolved_trace_id,
        request_id=request_id,
        tenant_id=tenant_id,
    )
    return resolved_trace_id


def finish_trace(trace_id: str | None = None) -> None:
    """Finish a trace and clear matching context variables.

    Args:
        trace_id (str | None): Trace to finish. When omitted, the current context
            trace is finished.

    Returns:
        None: Missing traces are ignored because observability must not affect the
            request outcome.
    """
    resolved_trace_id = trace_id or _current_trace_id.get()
    if resolved_trace_id is not None:
        _trace_store.finish_trace(resolved_trace_id)
    clear_trace_context(trace_id=resolved_trace_id)


def clear_trace_context(*, trace_id: str | None = None) -> None:
    """Clear trace-related context variables for the current task.

    Args:
        trace_id (str | None): Optional guard. When supplied, context is cleared
            only if it still points at that trace.

    Returns:
        None: Context variables are reset to their neutral values.
    """
    if trace_id is not None and _current_trace_id.get() != trace_id:
        return
    _current_trace_id.set(None)
    _current_request_id.set(None)
    _current_tenant_id.set(None)
    _current_span_id.set(None)


def current_trace_id() -> str | None:
    """Return the trace ID bound to the current async context.

    Args:
        None: The value is read from a context variable.

    Returns:
        str | None: Active trace ID, or None outside traced work.
    """
    return _current_trace_id.get()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[SpanRecord]:
    """Capture a redacted timed span under the active trace.

    Args:
        name (str): Logical operation name.
        **attributes (Any): Structured operational metadata. Values are treated
            as untrusted and redacted before storage/export.

    Returns:
        Iterator[SpanRecord]: Context manager yielding the mutable span record so
            callers can add final metadata before the span closes.
    """
    temporary_trace = False
    trace_id = _current_trace_id.get()
    request_id = _current_request_id.get() or "internal"
    tenant_id = _current_tenant_id.get() or 0
    if trace_id is None:
        temporary_trace = True
        trace_id = str(uuid.uuid4())
        start_trace(tenant_id=tenant_id, request_id=request_id, trace_id=trace_id)
    parent_span_id = _current_span_id.get()
    span_id = str(uuid.uuid4())
    span_token = _current_span_id.set(span_id)
    record = SpanRecord(
        trace_id=trace_id,
        request_id=request_id,
        name=name,
        attributes=redact_payload(attributes),
        started_at=time.time(),
        span_id=span_id,
        parent_span_id=parent_span_id,
    )
    try:
        yield record
    except Exception as exc:
        record.error = type(exc).__name__
        raise
    finally:
        record.ended_at = time.time()
        record.attributes = redact_payload(record.attributes)
        _current_span_id.reset(span_token)
        _trace_store.add_span(record, tenant_id=tenant_id)
        _langfuse_client.emit_span(record)
        if temporary_trace:
            finish_trace(trace_id)


def token_count(text: str) -> int:
    """Estimate token count using a deterministic whitespace split.

    Args:
        text (str): Prompt or response text to estimate.

    Returns:
        int: Number of non-empty whitespace-delimited tokens.
    """
    return len([token for token in text.split() if token])


def estimate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1k: float,
    output_cost_per_1k: float,
) -> float:
    """Estimate model call cost from token counts and configured rates.

    Args:
        input_tokens (int): Estimated prompt token count.
        output_tokens (int): Estimated completion token count.
        input_cost_per_1k (float): Cost per thousand prompt tokens.
        output_cost_per_1k (float): Cost per thousand completion tokens.

    Returns:
        float: Estimated cost in US dollars.
    """
    return (input_tokens / 1000.0 * input_cost_per_1k) + (
        output_tokens / 1000.0 * output_cost_per_1k
    )


def _datetime_from_timestamp(value: float) -> datetime:
    """Convert a Unix timestamp into a timezone-aware UTC datetime.

    Args:
        value (float): Unix timestamp in seconds.

    Returns:
        datetime: UTC datetime suitable for telemetry SDKs that reject floats.
    """
    return datetime.fromtimestamp(value, UTC)


def _record_observability_failure(reason: str, exc: Exception) -> None:
    """Record an observability subsystem failure without exposing sensitive data.

    Args:
        reason (str): Short failure category such as `langfuse_export`.
        exc (Exception): Exception raised by the subsystem.

    Returns:
        None: Metrics/logging failures are swallowed to avoid recursion.
    """
    try:
        from cafe_assistant.observability.metrics import record_quality_event

        record_quality_event(
            "observability_failures_total",
            reason=reason,
            error_type=type(exc).__name__,
        )
    except Exception:  # noqa: BLE001 - metrics failure cannot recurse into tracing.
        pass
    logger.warning("Observability failure: %s (%s)", reason, type(exc).__name__)


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    """Return a mapping value or an empty mapping for malformed payloads.

    Args:
        value (Any): Value loaded from JSON.

    Returns:
        Mapping[str, Any]: Original mapping when possible, otherwise `{}`.
    """
    return value if isinstance(value, Mapping) else {}


def _optional_float(value: Any) -> float | None:
    """Parse an optional float from a durable payload value.

    Args:
        value (Any): Raw JSON scalar.

    Returns:
        float | None: Parsed float, or None when the value is absent or invalid.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_value(value: Any, *, default: float) -> float:
    """Parse a float with a default for malformed durable payload values.

    Args:
        value (Any): Raw JSON scalar.
        default (float): Value returned when parsing fails.

    Returns:
        float: Parsed float or the supplied default.
    """
    parsed = _optional_float(value)
    return default if parsed is None else parsed
