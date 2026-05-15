from src.generators.grounded_qa import (
    build_expected_answer_outline,
    build_expected_behavior,
    generate_grounded_qa_cases,
    sentence_to_question,
)
from src.schemas import KnowledgeRule, RiskLevel, RuleType, TestType


def make_sample_rule() -> KnowledgeRule:
    return KnowledgeRule(
        rule_id="refund_policy_0001_rule_001",
        source_chunk_id="refund_policy_0001",
        source="refund_policy.md",
        section="Late Delivery Refunds",
        rule_type=RuleType.ELIGIBILITY,
        rule_text=(
            "Customers are eligible for a shipping-fee refund if delivery is delayed "
            "by more than 7 days and the customer has not already received compensation."
        ),
        condition=(
            "delivery is delayed by more than 7 days and the customer has not already "
            "received compensation"
        ),
        expected_action="Customers are eligible for a shipping-fee refund",
        required_evidence=["delay_days", "compensation_status"],
        risk_level=RiskLevel.MEDIUM,
    )


def test_sentence_to_question_for_late_delivery_refund():
    rule = make_sample_rule()

    question = sentence_to_question(rule)

    assert "package arrived more than 7 days late" in question
    assert "shipping-fee refund" in question


def test_build_expected_answer_outline():
    rule = make_sample_rule()

    outline = build_expected_answer_outline(rule)

    assert "Customers are eligible for a shipping-fee refund" in outline
    assert any("Condition:" in item for item in outline)
    assert any("delay_days" in item for item in outline)


def test_build_expected_behavior_for_eligibility():
    rule = make_sample_rule()

    behavior = build_expected_behavior(rule)

    assert "eligibility" in behavior.lower()
    assert "cite" in behavior.lower()


def test_generate_grounded_qa_cases():
    rule = make_sample_rule()

    cases = generate_grounded_qa_cases(
        rules=[rule],
        project_id="support_demo",
        dataset_version="v0.1.0",
    )

    assert len(cases) == 1

    case = cases[0]

    assert case.test_type == TestType.GROUNDED_QA
    assert case.risk_level == RiskLevel.MEDIUM
    assert case.project_id == "support_demo"
    assert case.dataset_version == "v0.1.0"
    assert len(case.required_citations) == 1
    assert case.required_citations[0].chunk_id == "refund_policy_0001"
    assert "grounded_answer" in case.tags
    assert case.metadata["source_rule_id"] == "refund_policy_0001_rule_001"