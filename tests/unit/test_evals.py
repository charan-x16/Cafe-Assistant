"""Unit tests for evaluation scoring gates.

These tests exercise the eval harness itself, not the retrieval or chat agent. They
make sure unsafe recommendations fail the hard gate, empty-safe-set labels are
enforced, and strict mode turns non-allergen family failures into release-blocking
exit codes.
"""

from __future__ import annotations

from cafe_assistant.agent.state_machine import ChatAgentResult
from cafe_assistant.domain.dietary import MenuItemView
from evals.allergen_safety import evaluate_results
from evals.common import (
    BTB_DATASET_PATH,
    EvalCase,
    EvalRunResult,
    load_cases,
    parse_response_item_names,
)
from evals.run_evals import EvalFamilyReport, _empty_safe_set_report, exit_code_for_reports


def test_allergen_gate_fails_when_unsafe_item_is_structured_safe_item() -> None:
    """Verify an unsafe structured recommendation fails the hard gate.

    Args:
        None.

    Returns:
        None:
            Failed assertions indicate the allergen scorer allowed a false
            negative through `ChatAgentResult.safe_items`.
    """

    result = _run_result(
        unsafe_item_names=["Peanut Butter Cookie"],
        safe_items=[_menu_item("Peanut Butter Cookie")],
        response="I can suggest something else.",
        menu_names={"Peanut Butter Cookie"},
    )

    report = evaluate_results([result])

    assert not report.passed
    assert report.false_negative_count == 1
    assert "Peanut Butter Cookie" in report.failures[0]


def test_allergen_gate_fails_when_unsafe_item_is_only_in_response_text() -> None:
    """Verify normalized response parsing catches unsafe text mentions.

    Args:
        None.

    Returns:
        None:
            Failed assertions indicate the scorer missed an unsafe item because
            the model response changed case or punctuation.
    """

    result = _run_result(
        unsafe_item_names=["Peanut Butter Cookie"],
        safe_items=[],
        response="try the peanut butter cookie today",
        menu_names={"Peanut Butter Cookie"},
    )

    report = evaluate_results([result])

    assert not report.passed
    assert report.false_negative_count == 1
    assert "Peanut Butter Cookie" in report.failures[0]


def test_response_parser_normalizes_accents_case_and_punctuation() -> None:
    """Verify item-name parsing is resilient to model text variations.

    Args:
        None.

    Returns:
        None:
            Failed assertions indicate safety scoring could miss a menu item
            mention because of capitalization, accents, or punctuation.
    """

    menu_names = {"Jalapeno Garlic Bread", "Cappuccino (Hot)"}

    parsed = parse_response_item_names(
        "I can suggest cappuccino hot, but not jalape\u00f1o garlic bread.",
        menu_names,
    )

    assert parsed == {"Jalapeno Garlic Bread", "Cappuccino (Hot)"}


def test_empty_safe_set_report_fails_when_items_are_returned() -> None:
    """Verify cases labeled empty-safe-set cannot return recommendations.

    Args:
        None.

    Returns:
        None:
            Failed assertions indicate the eval runner ignored the
            `expect_empty_safe_set` label.
    """

    result = _run_result(
        unsafe_item_names=[],
        safe_items=[_menu_item("Espresso")],
        response="I can suggest Espresso.",
        menu_names={"Espresso"},
        expect_empty_safe_set=True,
    )

    report = _empty_safe_set_report([result])

    assert not report.passed
    assert "fixture:case" in report.details
    assert "Espresso" in report.details


def test_exit_code_defaults_to_hard_allergen_gate_only() -> None:
    """Verify default eval exit behavior is controlled by allergen safety.

    Args:
        None.

    Returns:
        None:
            Failed assertions indicate normal CI mode would fail for a non-hard
            quality family instead of only the allergen gate.
    """

    reports = [
        EvalFamilyReport("allergen_safety", True, "ok"),
        EvalFamilyReport("relevance", False, "failure"),
    ]

    assert exit_code_for_reports(reports, strict=False) == 0
    assert exit_code_for_reports(reports, strict=True) == 1


def test_btb_eval_dataset_loads_real_catalog_cases() -> None:
    """Verify the BTB eval dataset is present and source-labeled.

    Args:
        None.

    Returns:
        None:
            Failed assertions indicate the production catalog cases are missing
            from the eval harness.
    """

    cases = load_cases(BTB_DATASET_PATH, source="btb_catalog")

    assert len(cases) >= 8
    assert any(case.id == "btb_tree_nut_pesto_pizza" for case in cases)
    assert {case.source for case in cases} == {"btb_catalog"}


def _run_result(
    *,
    unsafe_item_names: list[str],
    safe_items: list[MenuItemView],
    response: str,
    menu_names: set[str],
    expect_empty_safe_set: bool = False,
) -> EvalRunResult:
    """Build a minimal eval result for scorer unit tests.

    Args:
        unsafe_item_names (list[str]): Unsafe labels attached to the test case.
        safe_items (list[MenuItemView]): Structured recommendations returned by
            the fake agent result.
        response (str): Customer-facing response text to parse.
        menu_names (set[str]): Authoritative menu names for parser matching.
        expect_empty_safe_set (bool): Whether the test case should require no
            recommendations.

    Returns:
        EvalRunResult:
            A complete eval result object that can be scored by eval family
            helpers without running the real chat agent.
    """

    case = EvalCase(
        id="case",
        query="query",
        unsafe_item_names=unsafe_item_names,
        expected_any_item_names=[],
        expect_empty_safe_set=expect_empty_safe_set,
        expect_medical_refusal=False,
        source="fixture",
    )
    return EvalRunResult(
        case=case,
        result=ChatAgentResult(response=response, state_history=[], safe_items=safe_items),
        latency_ms=1.0,
        menu_names=menu_names,
        model_messages=[],
    )


def _menu_item(name: str) -> MenuItemView:
    """Create a minimal menu item view for eval scorer tests.

    Args:
        name (str): Item name to place in the structured recommendation list.

    Returns:
        MenuItemView:
            A complete, allergen-empty item view suitable for tests that focus on
            evaluation scoring rather than dietary filtering.
    """

    return MenuItemView(
        id=1,
        name=name,
        allergen_codes=set(),
        dietary_tags=set(),
        allergen_data_complete=True,
        sugar_grams=None,
    )
