from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

_ALLOWED_QR_KEYS = frozenset({"cafe_id", "location_id", "table_id"})
_TABLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class InvalidQrPayloadError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TenantContext:
    tenant_id: int
    location_id: int
    table_id: str


def parse_tenant_context(payload: str | Mapping[str, object]) -> TenantContext:
    raw = _coerce_payload(payload)
    keys = set(raw)
    if keys != _ALLOWED_QR_KEYS:
        extra = sorted(keys - _ALLOWED_QR_KEYS)
        missing = sorted(_ALLOWED_QR_KEYS - keys)
        raise InvalidQrPayloadError(
            f"QR payload must contain only cafe_id, location_id, and table_id. "
            f"Extra={extra}; missing={missing}."
        )

    table_id = str(raw["table_id"]).strip()
    if not _TABLE_ID_PATTERN.fullmatch(table_id):
        raise InvalidQrPayloadError("QR table_id is invalid.")

    return TenantContext(
        tenant_id=_positive_int(raw["cafe_id"], "cafe_id"),
        location_id=_positive_int(raw["location_id"], "location_id"),
        table_id=table_id,
    )


def _coerce_payload(payload: str | Mapping[str, object]) -> dict[str, object]:
    if isinstance(payload, Mapping):
        return dict(payload)

    text = payload.strip()
    if not text:
        raise InvalidQrPayloadError("QR payload is empty.")

    if text.startswith("{"):
        decoded = json.loads(text)
        if not isinstance(decoded, dict):
            raise InvalidQrPayloadError("QR JSON payload must be an object.")
        return dict(decoded)

    parsed_url = urlparse(text)
    query = parsed_url.query if parsed_url.query else text
    parsed_query = parse_qs(query, keep_blank_values=True, strict_parsing=False)
    return {
        key: values[-1] if values else ""
        for key, values in parsed_query.items()
    }


def _positive_int(value: object, field_name: str) -> int:
    try:
        parsed = int(str(value))
    except ValueError as exc:
        raise InvalidQrPayloadError(f"QR {field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise InvalidQrPayloadError(f"QR {field_name} must be a positive integer.")
    return parsed
