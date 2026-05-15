import pytest

from src.schemas import Citation, EvalCase, RiskLevel, TestType


def test_grounded_case_requires_citation():
    with pytest.raises(ValueError):
        EvalCase(
            test_id="bad_case_001",
            test_type=TestType.GROUNDED_QA,
            risk_level=RiskLevel.MEDIUM,
            user_query="Can I get a refund?",
            expected_behavior="Answer using refund policy.",
            expected_answer_outline=["Explain refund eligibility."],
            required_citations=[],
        )


def test_valid_grounded_case_passes():
    case = EvalCase(
        test_id="refund_grounded_001",
        test_type=TestType.GROUNDED_QA,
        risk_level=RiskLevel.MEDIUM,
        user_query="My package arrived 9 days late. Can I get my shipping fee back?",
        expected_behavior="Answer using late-delivery refund policy.",
        expected_answer_outline=[
            "Customer may be eligible for a shipping-fee refund.",
            "Delay must be more than 7 days.",
            "Customer must not already have received compensation.",
        ],
        required_citations=[
            Citation(
                chunk_id="refund_policy_0001",
                source="refund_policy.md",
                required_evidence="Late delivery refund rule.",
            )
        ],
    )

    assert case.test_id == "refund_grounded_001"
    assert case.test_type == TestType.GROUNDED_QA
    assert len(case.required_citations) == 1