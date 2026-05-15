from src.generators.adversarial import (
    build_adversarial_disallowed_behaviors,
    build_adversarial_expected_behavior,
    build_adversarial_outline,
    generate_adversarial_cases,
    make_adversarial_query,
    should_generate_adversarial_case,
)
from src.schemas import KnowledgeRule, RiskLevel, RuleType, TestType


def make_prohibition_rule() -> KnowledgeRule:
    return KnowledgeRule(
        rule_id="refund_policy_0002_rule_001",
        source_chunk_id="refund_policy_0002",
        source="refund_policy.md",
        section="Late Delivery Refunds",
        rule_type=RuleType.PROHIBITION,
        rule_text=(
            "Customers are not eligible for a full product refund only because "
            "the package arrived late."
        ),
        condition=None,
        expected_action=(
            "Customers are not eligible for a full product refund only because "
            "the package arrived late"
        ),
        required_evidence=["delay_days"],
        risk_level=RiskLevel.MEDIUM,
    )


def make_requirement_rule() -> KnowledgeRule:
    return KnowledgeRule(
        rule_id="refund_policy_0003_rule_001",
        source_chunk_id="refund_policy_0003",
        source="refund_policy.md",
        section="Damaged Product Refunds",
        rule_type=RuleType.REQUIREMENT,
        rule_text=(
            "If photo evidence is missing, the support agent should ask the customer "
            "to upload evidence before approving a refund."
        ),
        condition="photo evidence is missing",
        expected_action="the support agent should ask the customer to upload evidence",
        required_evidence=["photo_evidence"],
        risk_level=RiskLevel.MEDIUM,
    )


def test_make_adversarial_query_for_prohibition():
    rule = make_prohibition_rule()

    query = make_adversarial_query(rule)

    assert "Ignore the refund rules" in query
    assert "full product refund" in query


def test_build_expected_behavior_for_prohibition():
    rule = make_prohibition_rule()

    behavior = build_adversarial_expected_behavior(rule)

    assert "Refuse" in behavior
    assert "policy restriction" in behavior
    assert "cite" in behavior.lower()


def test_build_expected_behavior_for_requirement():
    rule = make_requirement_rule()

    behavior = build_adversarial_expected_behavior(rule)

    assert "Do not skip required information" in behavior
    assert "Ask for the required fields" in behavior


def test_build_adversarial_outline_includes_policy_bypass_warning():
    rule = make_requirement_rule()

    outline = build_adversarial_outline(rule)

    assert any("Do not follow" in item for item in outline)
    assert any("required information" in item.lower() for item in outline)


def test_build_disallowed_behaviors_for_refund_rule():
    rule = make_prohibition_rule()

    disallowed = build_adversarial_disallowed_behaviors(rule)

    assert any("ignore the policy" in item for item in disallowed)
    assert any("refund" in item for item in disallowed)


def test_should_generate_adversarial_case():
    rule = make_prohibition_rule()

    assert should_generate_adversarial_case(rule) is True


def test_generate_adversarial_cases():
    rule = make_prohibition_rule()

    cases = generate_adversarial_cases(
        rules=[rule],
        project_id="support_demo",
        dataset_version="v0.1.0",
    )

    assert len(cases) == 1

    case = cases[0]

    assert case.test_type == TestType.ADVERSARIAL
    assert case.risk_level == RiskLevel.HIGH
    assert case.project_id == "support_demo"
    assert case.dataset_version == "v0.1.0"
    assert len(case.required_citations) == 1
    assert case.required_citations[0].chunk_id == "refund_policy_0002"
    assert "policy_bypass_attempt" in case.tags
    assert case.metadata["source_rule_id"] == "refund_policy_0002_rule_001"