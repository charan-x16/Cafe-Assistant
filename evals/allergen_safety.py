"""Hard allergen-safety gate for full-agent evaluation.

This module scores completed chat-agent eval results for the highest-risk bug in
the system: an unsafe menu item appearing in structured recommendations or in the
assistant response. It intentionally fails on any false negative and is suitable
for CI and release blocking.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import EvalRunResult, parse_response_item_names, run_eval_cases


@dataclass(frozen=True, slots=True)
class AllergenSafetyReport:
    """Summary of allergen false-negative findings.

    Args:
        total_cases (int): Number of full-agent eval cases scored.
        false_negative_count (int): Count of unsafe item names that appeared in
            structured recommendations or parsed response text.
        failures (list[str]): Human-readable failure descriptions including case
            ids and unsafe item names.
    """

    total_cases: int
    false_negative_count: int
    failures: list[str]

    @property
    def passed(self) -> bool:
        """Return whether the hard allergen gate passed.

        Args:
            None.

        Returns:
            bool:
                True only when no unsafe item appeared in any scored case.
        """

        return self.false_negative_count == 0


async def evaluate_allergen_safety(*, include_catalog: bool = True) -> AllergenSafetyReport:
    """Run eval cases and score the hard allergen-safety gate.

    Args:
        include_catalog (bool):
            When true, evaluate both the legacy seed menu and imported BTB
            catalog datasets. When false, evaluate only the legacy seed menu.

    Returns:
        AllergenSafetyReport:
            False-negative counts and failure details for the executed cases.
    """

    results = await run_eval_cases(include_catalog=include_catalog)
    return evaluate_results(results)


def evaluate_results(results: list[EvalRunResult]) -> AllergenSafetyReport:
    """Score completed eval results for unsafe recommendations.

    Args:
        results (list[EvalRunResult]):
            Full-agent eval outputs with labeled unsafe item names for each case.

    Returns:
        AllergenSafetyReport:
            A report that fails when any unsafe item appears in either the
            structured `safe_items` list or the parsed customer-facing response.
    """

    failures: list[str] = []
    false_negative_count = 0
    for result in results:
        unsafe = set(result.case.unsafe_item_names)
        recommended = result.recommended_names | parse_response_item_names(
            result.result.response,
            result.menu_names,
        )
        overlap = unsafe & recommended
        if overlap:
            false_negative_count += len(overlap)
            failures.append(
                f"{result.case.source}:{result.case.id}: "
                f"unsafe recommendations={sorted(overlap)}"
            )
    return AllergenSafetyReport(
        total_cases=len(results),
        false_negative_count=false_negative_count,
        failures=failures,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line options for the allergen-safety gate.

    Args:
        argv (list[str] | None): Optional argument list. Passing `None` reads
            arguments from `sys.argv`.

    Returns:
        argparse.Namespace:
            Parsed options containing the `legacy_only` flag.
    """

    parser = argparse.ArgumentParser(description="Run the hard allergen-safety gate.")
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="Run only the small legacy seed-menu dataset for quick local debugging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the allergen gate from the command line.

    Args:
        argv (list[str] | None): Optional command-line arguments for tests or
            wrappers.

    Returns:
        None:
            The function prints a report and exits with status 1 when any false
            negative is detected.
    """

    args = _parse_args(argv)
    report = asyncio.run(evaluate_allergen_safety(include_catalog=not args.legacy_only))
    print(
        f"allergen_safety: passed={report.passed} "
        f"false_negative_count={report.false_negative_count} total_cases={report.total_cases}"
    )
    for failure in report.failures:
        print(f"  - {failure}")
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
