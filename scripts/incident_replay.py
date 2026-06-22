"""Command-line incident replay for local durable traces or the API endpoint.

Use local mode when running on the same host or shared volume as the trace JSONL
spool. Use API mode when replaying a trace from a deployed service; API mode
preserves the production tenant/admin authorization boundary.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from cafe_assistant.observability.replay import TraceNotFoundError, replay_trace


def main() -> None:
    """Parse CLI arguments, replay the requested trace, and print JSON.

    Args:
        None: Arguments are read from `sys.argv` by `argparse`.

    Returns:
        None: The replay payload is printed to stdout. Missing traces or failed
            API calls exit with a concise error message.
    """
    parser = argparse.ArgumentParser(description="Replay a redacted cafe-assistant trace.")
    parser.add_argument("trace_id", help="Trace ID from response headers, audit, or logs.")
    parser.add_argument(
        "--base-url",
        help="Optional API base URL. When omitted, the local durable trace spool is used.",
    )
    parser.add_argument(
        "--tenant-id",
        type=int,
        help="Tenant ID required when replaying through the API.",
    )
    parser.add_argument(
        "--admin-token",
        default=os.getenv("OBSERVABILITY_ADMIN_TOKEN"),
        help="Admin token for API replay. Defaults to OBSERVABILITY_ADMIN_TOKEN.",
    )
    args = parser.parse_args()

    if args.base_url:
        payload = _replay_from_api(
            base_url=args.base_url,
            trace_id=args.trace_id,
            tenant_id=args.tenant_id,
            admin_token=args.admin_token,
        )
    else:
        try:
            payload = replay_trace(args.trace_id)
        except TraceNotFoundError:
            raise SystemExit(f"Trace not found in local durable store: {args.trace_id}") from None
    print(json.dumps(payload, indent=2, default=str))


def _replay_from_api(
    *,
    base_url: str,
    trace_id: str,
    tenant_id: int | None,
    admin_token: str | None,
) -> dict[str, Any]:
    """Fetch a replay payload from the secured API endpoint.

    Args:
        base_url (str): API origin, for example `https://api.example.com`.
        trace_id (str): Trace ID to replay.
        tenant_id (int | None): Tenant scope required by the API dependency.
        admin_token (str | None): Observability admin token sent as `X-Admin-Token`.

    Returns:
        dict[str, Any]: Replay payload returned by the service.
    """
    if tenant_id is None:
        raise SystemExit("--tenant-id is required with --base-url")
    if not admin_token:
        raise SystemExit("--admin-token or OBSERVABILITY_ADMIN_TOKEN is required with --base-url")
    query = urlencode({"tenant_id": tenant_id})
    url = f"{base_url.rstrip('/')}/observability/replay/{trace_id}?{query}"
    request = Request(url, headers={"X-Admin-Token": admin_token})  # noqa: S310 - operator CLI.
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310 - explicit operator URL.
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        raise SystemExit(f"Replay API failed with HTTP {exc.code}: {exc.reason}") from exc
    return json.loads(body)


if __name__ == "__main__":
    main()
