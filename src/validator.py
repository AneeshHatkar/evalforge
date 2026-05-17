from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz

from src.citation_selector import validate_case_citations
from src.schemas import Chunk, EvalCase, TestType, ValidationResult


def normalize_query_for_duplicate_check(query: str) -> str:
    """
    Normalize a user query before duplicate comparison.
    """

    return " ".join(query.lower().strip().split())


def find_duplicate_test_ids(cases: List[EvalCase]) -> List[str]:
    """
    Find duplicate test IDs.
    """

    counts = Counter(case.test_id for case in cases)
    return [test_id for test_id, count in counts.items() if count > 1]


def find_near_duplicate_queries(
    cases: List[EvalCase],
    threshold: float = 0.92,
) -> List[Tuple[str, str, float]]:
    """
    Find near-duplicate user queries using fuzzy similarity.

    Returns:
        List of tuples:
        (test_id_a, test_id_b, similarity_score)
    """

    duplicates: List[Tuple[str, str, float]] = []

    for i in range(len(cases)):
        query_a = normalize_query_for_duplicate_check(cases[i].user_query)

        for j in range(i + 1, len(cases)):
            query_b = normalize_query_for_duplicate_check(cases[j].user_query)

            if not query_a or not query_b:
                continue

            score = fuzz.ratio(query_a, query_b) / 100.0

            if score >= threshold:
                duplicates.append((cases[i].test_id, cases[j].test_id, round(score, 4)))

    return duplicates


def validate_case_basic_fields(case: EvalCase) -> ValidationResult:
    """
    Validate basic non-citation quality requirements.
    """

    errors: List[str] = []
    warnings: List[str] = []

    if not case.test_id.strip():
        errors.append("Case is missing test_id.")

    if not case.user_query.strip():
        errors.append("Case is missing user_query.")

    if len(case.user_query.strip()) < 8:
        warnings.append("User query is very short.")

    if not case.expected_behavior.strip():
        errors.append("Case is missing expected_behavior.")

    if not case.expected_answer_outline:
        errors.append("Case is missing expected_answer_outline.")

    if case.expected_answer_outline:
        empty_outline_items = [
            index
            for index, item in enumerate(case.expected_answer_outline)
            if not item or not item.strip()
        ]

        if empty_outline_items:
            errors.append(f"Expected answer outline has empty item(s): {empty_outline_items}")

    if not case.tags:
        warnings.append("Case has no tags.")

    if case.test_type == TestType.TOOL_USE:
        if case.tool_expectation is None:
            errors.append("Tool-use case is missing tool_expectation.")
        else:
            if not case.tool_expectation.expected_tool:
                errors.append("Tool-use case is missing expected_tool.")

            if not case.tool_expectation.expected_tool_arguments:
                warnings.append("Tool-use case has no expected tool arguments.")

            if (
                case.tool_expectation.should_call_tool
                and case.tool_expectation.should_ask_clarification_if_missing
            ):
                errors.append(
                    "Tool-use case cannot both call tool immediately and ask clarification for missing arguments."
                )

    if case.test_type != TestType.TOOL_USE and case.tool_expectation is not None:
        errors.append("Non-tool-use case should not include tool_expectation.")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_single_case(
    case: EvalCase,
    chunks: List[Chunk],
    citation_support_threshold: float = 0.35,
) -> ValidationResult:
    """
    Validate one EvalCase using basic checks plus citation checks.
    """

    basic_result = validate_case_basic_fields(case)
    citation_result = validate_case_citations(
        case=case,
        chunks=chunks,
        support_threshold=citation_support_threshold,
    )

    errors = basic_result.errors + citation_result.errors
    warnings = basic_result.warnings + citation_result.warnings

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_cases(
    cases: List[EvalCase],
    chunks: List[Chunk],
    citation_support_threshold: float = 0.35,
    duplicate_query_threshold: float = 0.92,
) -> Dict[str, ValidationResult]:
    """
    Validate all generated cases.

    This checks:
    - individual case validity
    - duplicate test IDs
    - near-duplicate queries
    """

    results: Dict[str, ValidationResult] = {}

    duplicate_ids = set(find_duplicate_test_ids(cases))
    near_duplicates = find_near_duplicate_queries(cases, threshold=duplicate_query_threshold)

    near_duplicate_map: Dict[str, List[str]] = defaultdict(list)

    for test_id_a, test_id_b, score in near_duplicates:
        message = f"Near-duplicate query with {test_id_b} similarity={score}"
        near_duplicate_map[test_id_a].append(message)

        reverse_message = f"Near-duplicate query with {test_id_a} similarity={score}"
        near_duplicate_map[test_id_b].append(reverse_message)

    for case in cases:
        result = validate_single_case(
            case=case,
            chunks=chunks,
            citation_support_threshold=citation_support_threshold,
        )

        errors = list(result.errors)
        warnings = list(result.warnings)

        if case.test_id in duplicate_ids:
            errors.append(f"Duplicate test_id detected: {case.test_id}")

        warnings.extend(near_duplicate_map.get(case.test_id, []))

        results[case.test_id] = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    return results


def split_valid_invalid_cases(
    cases: List[EvalCase],
    validation_results: Dict[str, ValidationResult],
) -> Tuple[List[EvalCase], List[EvalCase]]:
    """
    Split cases into valid and invalid lists.
    """

    valid_cases: List[EvalCase] = []
    invalid_cases: List[EvalCase] = []

    for case in cases:
        result = validation_results.get(case.test_id)

        if result is not None and result.is_valid:
            valid_cases.append(case)
        else:
            invalid_cases.append(case)

    return valid_cases, invalid_cases


def attach_validation_errors(
    cases: List[EvalCase],
    validation_results: Dict[str, ValidationResult],
) -> List[EvalCase]:
    """
    Return updated EvalCase objects with validation_errors populated.

    This helps the review UI show why a case needs attention.
    """

    updated_cases: List[EvalCase] = []

    for case in cases:
        result = validation_results.get(case.test_id)

        if result is None:
            updated_cases.append(case)
            continue

        updated_data = case.model_dump()
        updated_data["validation_errors"] = result.errors

        updated_cases.append(EvalCase(**updated_data))

    return updated_cases


def summarize_validation_results(
    cases: List[EvalCase],
    validation_results: Dict[str, ValidationResult],
) -> Dict[str, object]:
    """
    Build a benchmark-quality summary report.

    This will later appear in the Streamlit dashboard.
    """

    total_cases = len(cases)
    valid_count = sum(1 for result in validation_results.values() if result.is_valid)
    invalid_count = total_cases - valid_count

    warning_count = sum(len(result.warnings) for result in validation_results.values())
    error_count = sum(len(result.errors) for result in validation_results.values())

    test_type_counts = Counter(case.test_type.value for case in cases)
    risk_counts = Counter(case.risk_level.value for case in cases)
    review_status_counts = Counter(case.review_status.value for case in cases)

    cases_with_citations = sum(1 for case in cases if case.required_citations)

    citation_coverage = cases_with_citations / total_cases if total_cases else 0.0
    validity_rate = valid_count / total_cases if total_cases else 0.0

    invalid_case_ids = [
        test_id
        for test_id, result in validation_results.items()
        if not result.is_valid
    ]

    warning_case_ids = [
        test_id
        for test_id, result in validation_results.items()
        if result.warnings
    ]

    return {
        "total_cases": total_cases,
        "valid_cases": valid_count,
        "invalid_cases": invalid_count,
        "validity_rate": round(validity_rate, 4),
        "cases_with_citations": cases_with_citations,
        "citation_coverage": round(citation_coverage, 4),
        "total_errors": error_count,
        "total_warnings": warning_count,
        "test_type_distribution": dict(test_type_counts),
        "risk_distribution": dict(risk_counts),
        "review_status_distribution": dict(review_status_counts),
        "invalid_case_ids": invalid_case_ids,
        "warning_case_ids": warning_case_ids,
    }


def print_validation_report(summary: Dict[str, object]) -> None:
    """
    Print a readable validation summary in the terminal.
    """

    print("=" * 80)
    print("EvalForge Validation Report")
    print("=" * 80)

    print(f"Total cases:          {summary['total_cases']}")
    print(f"Valid cases:          {summary['valid_cases']}")
    print(f"Invalid cases:        {summary['invalid_cases']}")
    print(f"Validity rate:        {summary['validity_rate']}")
    print(f"Citation coverage:    {summary['citation_coverage']}")
    print(f"Total errors:         {summary['total_errors']}")
    print(f"Total warnings:       {summary['total_warnings']}")

    print("\nTest type distribution:")
    for test_type, count in summary["test_type_distribution"].items():
        print(f"  - {test_type}: {count}")

    print("\nRisk distribution:")
    for risk_level, count in summary["risk_distribution"].items():
        print(f"  - {risk_level}: {count}")

    if summary["invalid_case_ids"]:
        print("\nInvalid case IDs:")
        for case_id in summary["invalid_case_ids"]:
            print(f"  - {case_id}")

    if summary["warning_case_ids"]:
        print("\nCase IDs with warnings:")
        for case_id in summary["warning_case_ids"]:
            print(f"  - {case_id}")