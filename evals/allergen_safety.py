from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import EvalRunResult, parse_response_item_names, run_eval_cases


@dataclass(frozen=True, slots=True)
class AllergenSafetyReport:
    total_cases: int
    false_negative_count: int
    failures: list[str]

    @property
    def passed(self) -> bool:
        return self.false_negative_count == 0


async def evaluate_allergen_safety() -> AllergenSafetyReport:
    results = await run_eval_cases()
    return evaluate_results(results)


def evaluate_results(results: list[EvalRunResult]) -> AllergenSafetyReport:
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
                f"{result.case.id}: unsafe recommendations={sorted(overlap)}"
            )
    return AllergenSafetyReport(
        total_cases=len(results),
        false_negative_count=false_negative_count,
        failures=failures,
    )


def main() -> None:
    report = asyncio.run(evaluate_allergen_safety())
    print(
        f"allergen_safety: passed={report.passed} "
        f"false_negative_count={report.false_negative_count} total_cases={report.total_cases}"
    )
    for failure in report.failures:
        print(f"  - {failure}")
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
