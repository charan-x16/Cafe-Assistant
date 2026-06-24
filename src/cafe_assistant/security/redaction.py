"""PII, health-data, and secret redaction for logs, audits, and traces.

Every user message, profile payload, provider credential, and retrieved document is
considered untrusted operational data. This module provides conservative helpers
used before payloads are written to audit rows, trace spans, or Python logging.
It intentionally redacts more than the minimum needed so phone numbers, health
facts, bearer tokens, cookies, passwords, and provider keys are not persisted in
raw form.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

_PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")
_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_AUTH_HEADER_PATTERN = re.compile(
    r"\b(?P<key>authorization|proxy-authorization)\b\s*[:=]\s*"
    r"(?:bearer\s+)?[^,'\"\s}\]]+",
    re.IGNORECASE,
)
_COOKIE_PATTERN = re.compile(
    r"\b(?P<key>cookie|set-cookie)\b\s*[:=]\s*[^\n\r,;]+",
    re.IGNORECASE,
)
_KEY_VALUE_SECRET_PATTERN = re.compile(
    r"\b(?P<key>[a-z0-9_-]*(?:api[_-]?key|secret|token|password|"
    r"auth[_-]?token)[a-z0-9_-]*)\b"
    r"\s*[:=]\s*['\"]?[^,'\"\s}\]]+",
    re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(r"\bbearer\s+[a-z0-9._~+/=-]+", re.IGNORECASE)
_HEALTH_PATTERN = re.compile(
    r"\b("
    r"allerg(?:y|ic|ies)|peanuts?|tree\s*nuts?|dairy|gluten|soy|eggs?|"
    r"diabet(?:es|ic)|insulin|blood\s*sugar|glucose|a1c|carb(?:s| counting)?"
    r")\b",
    re.IGNORECASE,
)
_IP_PATTERN = re.compile(
    r"\b(?P<key>client[_-]?ip|ip)\b\s*[:=]\s*"
    r"(?:\d{1,3}(?:\.\d{1,3}){3}|[a-f0-9:]{3,})",
    re.IGNORECASE,
)
_SENSITIVE_KEYS = {
    "allergies",
    "api_key",
    "auth_token",
    "authorization",
    "avoid_allergens",
    "bearer",
    "client_ip",
    "code",
    "cookie",
    "device_token",
    "dietary_facts",
    "email",
    "health",
    "identity_device_token_hash_secret",
    "ip",
    "langfuse_secret_key",
    "llm_api_key",
    "message",
    "openai_api_key",
    "password",
    "phone",
    "phone_number",
    "qdrant_api_key",
    "rate_limit_hash_secret",
    "secret",
    "session_id",
    "set_cookie",
    "token",
}
_original_record_factory: object | None = None


def redact_text(value: str) -> str:
    """Redact sensitive substrings from free-form text.

    Args:
        value (str):
            Log, trace, or audit text that may contain PII, health facts, or secrets.

    Returns:
        str:
            Text with recognized sensitive patterns replaced by redaction markers.
    """
    redacted = _AUTH_HEADER_PATTERN.sub(lambda match: f"{match.group('key')}=[REDACTED]", value)
    redacted = _COOKIE_PATTERN.sub(lambda match: f"{match.group('key')}=[REDACTED]", redacted)
    redacted = _KEY_VALUE_SECRET_PATTERN.sub(
        lambda match: f"{match.group('key')}=[REDACTED]",
        redacted,
    )
    redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", redacted)
    redacted = _PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    redacted = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
    redacted = _IP_PATTERN.sub(
        lambda match: f"{match.group('key')}=[REDACTED_IP]",
        redacted,
    )
    redacted = _HEALTH_PATTERN.sub("[REDACTED_HEALTH]", redacted)
    return redacted


def redact_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Redact a JSON-like mapping before persistence or emission.

    Args:
        payload (Mapping[str, Any] | None):
            Structured payload from audit, tracing, logging, or metrics code.

    Returns:
        dict[str, Any]:
            Copy of the payload with sensitive keys and string values redacted.
    """
    if payload is None:
        return {}
    return {str(key): _redact_value(str(key), value) for key, value in payload.items()}


class RedactingFilter(logging.Filter):
    """Logging filter that redacts records before handlers format them."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact a log record message and positional arguments in place.

        Args:
            record (logging.LogRecord):
                Log record about to be emitted by Python logging.

        Returns:
            bool:
                Always True so the record continues through normal logging flow.
        """
        record.msg = redact_text(str(record.msg))
        if record.args:
            record.args = tuple(
                redact_text(str(arg)) if isinstance(arg, str) else _redact_value("", arg)
                for arg in record.args
            )
        return True


def configure_redacted_logging() -> None:
    """Install redaction on the root logger and future log records.

    Args:
        None:
            Logging configuration is process global.

    Returns:
        None:
            Repeated calls are idempotent and do not stack duplicate filters.
    """
    global _original_record_factory
    root = logging.getLogger()
    if any(isinstance(filter_, RedactingFilter) for filter_ in root.filters):
        return
    root.addFilter(RedactingFilter())
    for handler in root.handlers:
        handler.addFilter(RedactingFilter())
    if _original_record_factory is None:
        _original_record_factory = logging.getLogRecordFactory()

        def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = _original_record_factory(*args, **kwargs)  # type: ignore[misc]
            RedactingFilter().filter(record)
            return record

        logging.setLogRecordFactory(factory)


def _redact_value(key: str, value: Any) -> Any:
    """Redact one structured value using its key and runtime type.

    Args:
        key (str):
            Mapping key associated with the value, if known.
        value (Any):
            Arbitrary structured value to redact recursively.

    Returns:
        Any:
            Redacted scalar or recursively redacted container.
    """
    normalized_key = key.lower().replace("-", "_")
    if normalized_key in _SENSITIVE_KEYS:
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return redact_payload(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_redact_value(key, item) for item in value]
    return value
