"""Import By The Brew source Markdown into the production catalog schema.
Parses configured source files, writes a versioned catalog into Postgres/Neon, and prints a
setup-friendly import summary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from cafe_assistant.db.session import async_session_maker
from cafe_assistant.ingestion.btb_markdown import (
    BTB_SOURCE_DIR,
    import_btb_documents,
    import_result_to_dict,
)


async def _run(source_dir: Path, publish: bool) -> dict[str, object]:
    """Run the requested value.

    Args:
        source_dir (Path):
            Source dir value required to perform this operation.
        publish (bool):
            Publish value required to perform this operation.

    Returns:
        dict[str, object]:
            Value produced for the caller according to the function contract.
    """
    started_at = time.perf_counter()
    print("Importing BTB documents into the catalog...", flush=True)
    async with async_session_maker() as session:
        result = await import_btb_documents(
            session,
            source_dir=source_dir,
            publish=publish,
        )
    elapsed_seconds = time.perf_counter() - started_at
    result_payload = import_result_to_dict(result)
    result_payload["elapsed_seconds"] = round(elapsed_seconds, 2)
    return result_payload


def main() -> None:
    """Run the command-line interface for this module.

    Args:
        None.

    Returns:
        None:
            No value is returned; the function completes through side effects or validation.
    """
    parser = argparse.ArgumentParser(
        description="Import By The Brew Markdown source documents into the production catalog."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=BTB_SOURCE_DIR,
        help="Directory containing the BTB Markdown source documents.",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Import as staged instead of publishing the menu version.",
    )
    args = parser.parse_args()

    result = asyncio.run(_run(args.source_dir, publish=not args.staged))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()