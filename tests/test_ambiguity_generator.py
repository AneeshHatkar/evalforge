from src.generators.ambiguity import (
    build_ambiguity_expected_behavior,
    build_ambiguity_outline,
    build_missing_information,
    evidence_field_to_question,
    generate_ambiguity_cases,
    make_ambiguous_query,
)
from src.schemas import KnowledgeRule, RiskLevel, RuleType, TestType


def make_late_delivery_rule() -> KnowledgeRule:
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


def test_evidence_field_to_question_known_field():
    assert evidence_field_to_question("order_id") == "What is your order ID?"
    assert evidence_field_to_question("delay_days") == "How many days late was the delivery?"


def test_evidence_field_to_question_unknown_field():
    question = evidence_field_to_question("custom_field")
    assert "custom_field" in question


def test_build_missing_information_from_required_evidence():
    rule = make_late_delivery_rule()

    missing = build_missing_information(rule)

    assert missing == ["delay_days", "compensation_status"]


def test_make_ambiguous_query_for_late_refund():
    rule = make_late_delivery_rule()

    query = make_ambiguous_query(rule)

    assert "refund" in query.lower()
    assert "late package" in query.lower()


def test_build_ambiguity_expected_behavior_mentions_missing_fields():
    rule = make_late_delivery_rule()
    missing_fields = ["delay_days", "compensation_status"]

    behavior = build_ambiguity_expected_behavior(rule, missing_fields)

    assert "Do not make a final policy decision" in behavior
    assert "delay_days" in behavior
    assert "compensation_status" in behavior


def test_build_ambiguity_outline_includes_questions():
    rule = make_late_delivery_rule()
    missing_fields = ["delay_days", "compensation_status"]

    outline = build_ambiguity_outline(rule, missing_fields)

    assert any("How many days late" in item for item in outline)
    assert any("already received compensation" in item for item in outline)


def test_generate_ambiguity_cases():
    rule = make_late_delivery_rule()

    cases = generate_ambiguity_cases(
        rules=[rule],
        project_id="support_demo",
        dataset_version="v0.1.0",
    )

    assert len(cases) == 1

    case = cases[0]

    assert case.test_type == TestType.AMBIGUITY
    assert case.project_id == "support_demo"
    assert case.dataset_version == "v0.1.0"
    assert case.risk_level == RiskLevel.MEDIUM
    assert len(case.required_citations) == 1
    assert case.required_citations[0].chunk_id == "refund_policy_0001"
    assert "clarification_required" in case.tags
    assert case.metadata["missing_fields"] == ["delay_days", "compensation_status"]