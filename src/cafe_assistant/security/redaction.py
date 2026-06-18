from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

_PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")
_TOKEN_PATTERN = re.compile(
    r"\b(?P<key>device[_-]?token|token|secret|api[_-]?key|password)\b"
    r"\s*[:=]\s*['\"]?[^,'\"\s}]+",
    re.IGNORECASE,
)
_HEALTH_PATTERN = re.compile(
    r"\b("
    r"allerg(?:y|ic|ies)|peanuts?|tree\s*nuts?|dairy|gluten|soy|eggs?|"
    r"diabet(?:es|ic)|insulin|blood\s*sugar|glucose|a1c|carb(?:s| counting)?"
    r")\b",
    re.IGNORECASE,
)
_SENSITIVE_KEYS = {
    "phone",
    "phone_number",
    "phone_hash",
    "device_token",
    "token",
    "secret",
    "password",
    "api_key",
    "dietary_facts",
    "avoid_allergens",
    "allergies",
    "health",
    "message",
}
_original_record_factory: object | None = None


def redact_text(value: str) -> str:
    redacted = _PHONE_PATTERN.sub("[REDACTED_PHONE]", value)
    redacted = _TOKEN_PATTERN.sub(lambda match: f"{match.group('key')}=[REDACTED]", redacted)
    redacted = _HEALTH_PATTERN.sub("[REDACTED_HEALTH]", redacted)
    return redacted


def redact_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    return {
        str(key): _redact_value(str(key), value)
        for key, value in payload.items()
    }


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(str(record.msg))
        if record.args:
            record.args = tuple(
                redact_text(str(arg)) if isinstance(arg, str) else _redact_value("", arg)
                for arg in record.args
            )
        return True


def configure_redacted_logging() -> None:
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
    if key.lower() in _SENSITIVE_KEYS:
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return redact_payload(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_redact_value(key, item) for item in value]
    return value
