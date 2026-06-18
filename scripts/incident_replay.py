from __future__ import annotations

import argparse
import json

from cafe_assistant.observability.replay import TraceNotFoundError, replay_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a stored in-process trace.")
    parser.add_argument("trace_id")
    args = parser.parse_args()
    try:
        payload = replay_trace(args.trace_id)
    except TraceNotFoundError:
        raise SystemExit(f"Trace not found: {args.trace_id}") from None
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
