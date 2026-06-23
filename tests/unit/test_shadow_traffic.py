"""Unit tests for remote shadow-traffic release checks.

The shadow-traffic script runs against deployed canaries, so its scoring logic is
kept pure and tested here. These tests ensure release checks fail on unsafe item
leaks, missing medical refusal copy, and recommendations in cases labeled as an
empty safe set.
"""

from __future__ import annotations

from deploy.shadow_traffic import (
    ShadowCase,
    evaluate_shadow_response,
    load_shadow_cases,
)


def test_shadow_cases_load_legacy_and_btb_datasets() -> None:
    """Verify shadow traffic covers both legacy and BTB adversarial datasets.

    Args:
        None.

    Returns:
        None: Failed assertions indicate release shadow traffic lost dataset coverage.
    """

    cases = load_shadow_cases(["legacy", "btb"])

    assert len(cases) >= 20
    assert {case.dataset for case in cases} == {"legacy", "btb"}


def test_shadow_response_fails_on_normalized_unsafe_item_leak() -> None:
    """Verify unsafe item detection ignores case and accent differences.

    Args:
        None.

    Returns:
        None: Failed assertions indicate canary shadow traffic could miss a leak.
    """

    case = _case(unsafe_item_names=["Jalapeno Garlic Bread"])

    failures = evaluate_shadow_response(
        case,
        status_code=200,
        response_text="You can have jalape\u00f1o garlic bread today.",
    )

    assert [failure.reason for failure in failures] == ["unsafe_item_leak"]


def test_shadow_response_fails_empty_safe_set_recommendation() -> None:
    """Verify empty-safe-set cases cannot contain recommendation phrasing.

    Args:
        None.

    Returns:
        None: Failed assertions indicate release checks allow guessed recommendations.
    """

    case = _case(expect_empty_safe_set=True)

    failures = evaluate_shadow_response(
        case,
        status_code=200,
        response_text="I can suggest Espresso.",
    )

    assert {failure.reason for failure in failures} == {"empty_safe_set"}
    assert len(failures) == 2


def test_shadow_response_accepts_staff_check_empty_safe_set_fallback() -> None:
    """Verify empty-safe-set cases pass with uncertainty plus staff-check copy.

    Args:
        None.

    Returns:
        None: Failed assertions indicate valid safe fallback copy is rejected.
    """

    case = _case(expect_empty_safe_set=True)

    failures = evaluate_shadow_response(
        case,
        status_code=200,
        response_text="I can't confirm a safe option. Please check with cafe staff.",
    )

    assert failures == []


def test_shadow_response_fails_missing_medical_disclaimer() -> None:
    """Verify medical cases require the not-medical-advice note.

    Args:
        None.

    Returns:
        None: Failed assertions indicate medical refusal checks are too weak.
    """

    case = _case(expect_medical_refusal=True)

    failures = evaluate_shadow_response(
        case,
        status_code=200,
        response_text="Please ask your clinician.",
    )

    assert [failure.reason for failure in failures] == ["medical_refusal"]


def _case(
    *,
    unsafe_item_names: list[str] | None = None,
    expect_empty_safe_set: bool = False,
    expect_medical_refusal: bool = False,
) -> ShadowCase:
    """Build a minimal shadow case for scorer tests.

    Args:
        unsafe_item_names (list[str] | None): Unsafe labels attached to the case.
        expect_empty_safe_set (bool): Whether the response must avoid recommendations.
        expect_medical_refusal (bool): Whether the response must include medical refusal copy.

    Returns:
        ShadowCase: Complete case object accepted by `evaluate_shadow_response`.
    """

    return ShadowCase(
        id="case",
        dataset="fixture",
        query="query",
        unsafe_item_names=unsafe_item_names or [],
        expect_empty_safe_set=expect_empty_safe_set,
        expect_medical_refusal=expect_medical_refusal,
    )
