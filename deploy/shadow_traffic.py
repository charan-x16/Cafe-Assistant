from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx

DATASET = Path(__file__).parents[1] / "evals" / "datasets" / "agent_eval_cases.json"


async def main() -> None:
    target_url = os.environ.get("TARGET_URL")
    if not target_url:
        raise SystemExit("Set TARGET_URL, e.g. https://canary.example.com")

    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    failures: list[str] = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for case in cases:
            response = await client.post(
                f"{target_url.rstrip('/')}/chat",
                json={
                    "session_id": f"shadow-{case['id']}",
                    "tenant_id": int(os.environ.get("TENANT_ID", "1")),
                    "message": case["query"],
                },
            )
            text = response.text
            for unsafe_item in case["unsafe_item_names"]:
                if unsafe_item in text:
                    failures.append(f"{case['id']}: unsafe item leaked: {unsafe_item}")

    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        raise SystemExit(1)
    print(f"PASS shadow traffic: {len(cases)} cases")


if __name__ == "__main__":
    asyncio.run(main())
