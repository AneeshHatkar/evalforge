from src.schemas import (
    Chunk,
    Citation,
    EvalCase,
    RiskLevel,
    TestType,
    ToolExpectation,
)
from src.validator import (
    attach_validation_errors,
    find_duplicate_test_ids,
    find_near_duplicate_queries,
    split_valid_invalid_cases,
    summarize_validation_results,
    validate_case_basic_fields,
    validate_cases,
    validate_single_case,
)


def make_chunk() -> Chunk:
    return Chunk(
        chunk_id="refund_policy_0001",
        document_id="refund_policy",
        source="refund_policy.md",
        section="Late Delivery Refunds",
        text=(
            "Customers are eligible for a shipping-fee refund if delivery is delayed "
            "by more than 7 days and the customer has not already received compensation."
        ),
    )


def make_valid_grounded_case(test_id: str = "refund_policy_0001_grounded_001") -> EvalCase:
    return EvalCase(
        test_id=test_id,
        test_type=TestType.GROUNDED_QA,
        risk_level=RiskLevel.MEDIUM,
        user_query="Can I get a shipping refund if my package was more than 7 days late?",
        expected_behavior="Answer using the late-delivery refund policy and cite the source.",
        expected_answer_outline=[
            "Customer may be eligible for a shipping-fee refund.",
            "Delivery must be delayed by more than 7 days.",
            "Customer must not already have received compensation.",
        ],
        required_citations=[
            Citation(
                chunk_id="refund_policy_0001",
                source="refund_policy.md",
                required_evidence=(
                    "Customers are eligible for a shipping-fee refund if delivery is delayed "
                    "by more than 7 days and the customer has not already received compensation."
                ),
            )
        ],
        tags=["grounded_answer", "refund"],
    )


def test_find_duplicate_test_ids():
    cases = [
        make_valid_grounded_case("case_001"),
        make_valid_grounded_case("case_001"),
        make_valid_grounded_case("case_002"),
    ]

    duplicates = find_duplicate_test_ids(cases)

    assert duplicates == ["case_001"]


def test_find_near_duplicate_queries():
    case_a = make_valid_grounded_case("case_a")
    case_b = make_valid_grounded_case("case_b")

    case_b_data = case_b.model_dump()
    case_b_data["user_query"] = (
        "Can I get a shipping refund if my package was more than 7 days late?"
    )
    case_b = EvalCase(**case_b_data)

    duplicates = find_near_duplicate_queries([case_a, case_b], threshold=0.90)

    assert len(duplicates) == 1
    assert duplicates[0][0] == "case_a"
    assert duplicates[0][1] == "case_b"


def test_validate_case_basic_fields_valid_case():
    case = make_valid_grounded_case()

    result = validate_case_basic_fields(case)

    assert result.is_valid is True
    assert result.errors == []


def test_validate_case_basic_fields_tool_case_requires_expectation():
    # We cannot create an invalid TOOL_USE EvalCase directly because Pydantic
    # schema validation rejects it. So this test checks a valid tool case.
    case = EvalCase(
        test_id="tool_case_001",
        test_type=TestType.TOOL_USE,
        risk_level=RiskLevel.MEDIUM,
        user_query="Please process the shipping refund.",
        expected_behavior="Ask for missing order_id before calling the tool.",
        expected_answer_outline=["Relevant tool: issue_shipping_refund"],
        required_citations=[],
        tags=["tool_use"],
        tool_expectation=ToolExpectation(
            expected_tool="issue_shipping_refund",
            expected_tool_arguments={
                "order_id": "required_from_user_or_context",
                "refund_type": "shipping_fee_only",
            },
            should_call_tool=False,
            should_ask_clarification_if_missing=["order_id"],
        ),
    )

    result = validate_case_basic_fields(case)

    assert result.is_valid is True


def test_validate_single_case_valid():
    chunk = make_chunk()
    case = make_valid_grounded_case()

    result = validate_single_case(case, [chunk])

    assert result.is_valid is True


def test_validate_single_case_missing_chunk_invalid():
    case = make_valid_grounded_case()

    result = validate_single_case(case, [])

    assert result.is_valid is False
    assert any("missing chunk_id" in error for error in result.errors)


def test_validate_cases_duplicate_id_invalid():
    chunk = make_chunk()

    case_a = make_valid_grounded_case("duplicate_case")
    case_b = make_valid_grounded_case("duplicate_case")

    results = validate_cases([case_a, case_b], [chunk])

    assert results["duplicate_case"].is_valid is False
    assert any("Duplicate test_id" in error for error in results["duplicate_case"].errors)


def test_split_valid_invalid_cases():
    chunk = make_chunk()

    valid_case = make_valid_grounded_case("valid_case")
    invalid_case = make_valid_grounded_case("invalid_case")

    invalid_data = invalid_case.model_dump()
    invalid_data["required_citations"] = [
        {
            "chunk_id": "missing_chunk",
            "source": "refund_policy.md",
            "required_evidence": "missing evidence",
        }
    ]
    invalid_case = EvalCase(**invalid_data)

    cases = [valid_case, invalid_case]
    results = validate_cases(cases, [chunk])

    valid_cases, invalid_cases = split_valid_invalid_cases(cases, results)

    assert len(valid_cases) == 1
    assert len(invalid_cases) == 1
    assert valid_cases[0].test_id == "valid_case"
    assert invalid_cases[0].test_id == "invalid_case"


def test_attach_validation_errors():
    chunk = make_chunk()
    case = make_valid_grounded_case("invalid_case")

    case_data = case.model_dump()
    case_data["required_citations"] = [
        {
            "chunk_id": "missing_chunk",
            "source": "refund_policy.md",
            "required_evidence": "missing evidence",
        }
    ]
    case = EvalCase(**case_data)

    results = validate_cases([case], [chunk])
    updated_cases = attach_validation_errors([case], results)

    assert len(updated_cases[0].validation_errors) > 0


def test_summarize_validation_results():
    chunk = make_chunk()
    case = make_valid_grounded_case()

    results = validate_cases([case], [chunk])
    summary = summarize_validation_results([case], results)

    assert summary["total_cases"] == 1
    assert summary["valid_cases"] == 1
    assert summary["invalid_cases"] == 0
    assert summary["validity_rate"] == 1.0
    assert summary["citation_coverage"] == 1.0
    assert summary["test_type_distribution"]["grounded_policy_qa"] == 1