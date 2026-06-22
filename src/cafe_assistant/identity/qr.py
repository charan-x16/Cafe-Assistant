"""QR payload parsing for cafe, location, and table context.

QR codes are intentionally not identity-bearing credentials. They may contain
only cafe_id, location_id, and table_id values that stamp tenant/location/table
context at the API gateway. This module validates the payload shape and scalar
formats; database ownership validation happens in `api.deps` because it needs an
async database session.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

_ALLOWED_QR_KEYS = frozenset({"cafe_id", "location_id", "table_id"})
_TABLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class InvalidQrPayloadError(ValueError):
    """Raised when a QR payload has missing, extra, or malformed fields."""


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Parsed context values carried by a table QR code.

    Attributes:
        tenant_id (int):
            Cafe/tenant identifier from the QR cafe_id field.
        location_id (int):
            Location identifier from the QR location_id field.
        table_id (str):
            Non-secret table identifier used for in-cafe context.
    """

    tenant_id: int
    location_id: int
    table_id: str


def parse_tenant_context(payload: str | Mapping[str, object]) -> TenantContext:
    """Parse and validate a QR payload into tenant/location/table context.

    Args:
        payload (str | Mapping[str, object]):
            QR payload as a JSON object string, URL/query string, or mapping.
            The payload must contain exactly cafe_id, location_id, and table_id.

    Returns:
        TenantContext:
            Parsed positive integer tenant/location IDs and normalized table ID.
    """
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
    """Coerce supported QR encodings into a plain dictionary.

    Args:
        payload (str | Mapping[str, object]):
            Mapping, JSON object string, full URL, or query-string payload.

    Returns:
        dict[str, object]:
            Dictionary form used by `parse_tenant_context` validation.
    """
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
    """Parse one QR field as a positive integer.

    Args:
        value (object):
            Raw field value from the QR payload.
        field_name (str):
            Field name used in validation errors.

    Returns:
        int:
            Positive integer value.
    """
    try:
        parsed = int(str(value))
    except ValueError as exc:
        raise InvalidQrPayloadError(f"QR {field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise InvalidQrPayloadError(f"QR {field_name} must be a positive integer.")
    return parsed
