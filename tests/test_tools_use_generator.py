import json

from src.generators.tool_use import (
    build_tool_arguments,
    generate_tool_use_cases,
    infer_expected_tool_name,
    infer_missing_arguments,
    parse_tool_schema_text,
)
from src.schemas import KnowledgeRule, RiskLevel, RuleType, TestType


def sample_tool_schema_text() -> str:
    return json.dumps(
        {
            "tools": [
                {
                    "name": "issue_shipping_refund",
                    "description": "Issues a shipping-fee-only refund.",
                    "required_arguments": [
                        "order_id",
                        "refund_type",
                        "eligibility_reason",
                    ],
                },
                {
                    "name": "create_replacement_order",
                    "description": "Creates a replacement order.",
                    "required_arguments": [
                        "order_id",
                        "damage_evidence",
                        "replacement_reason",
                    ],
                },
                {
                    "name": "escalate_to_human",
                    "description": "Escalates a case to human review.",
                    "required_arguments": [
                        "order_id",
                        "escalation_reason",
                    ],
                },
            ]
        }
    )


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


def make_damaged_product_rule() -> KnowledgeRule:
    return KnowledgeRule(
        rule_id="refund_policy_0002_rule_001",
        source_chunk_id="refund_policy_0002",
        source="refund_policy.md",
        section="Damaged Product Refunds",
        rule_type=RuleType.ELIGIBILITY,
        rule_text=(
            "Customers may request a replacement or refund if the product arrives damaged "
            "and they provide photo evidence within 14 days of delivery."
        ),
        condition="the product arrives damaged and they provide photo evidence within 14 days of delivery",
        expected_action="Customers may request a replacement or refund",
        required_evidence=["photo_evidence"],
        risk_level=RiskLevel.MEDIUM,
    )


def make_escalation_rule() -> KnowledgeRule:
    return KnowledgeRule(
        rule_id="shipping_policy_0003_rule_001",
        source_chunk_id="shipping_policy_0003",
        source="shipping_policy.md",
        section="Lost Packages",
        rule_type=RuleType.ESCALATION,
        rule_text=(
            "If tracking shows no movement for more than 10 days, the support agent "
            "should escalate the case for lost-package investigation."
        ),
        condition="tracking shows no movement for more than 10 days",
        expected_action="the support agent should escalate the case for lost-package investigation",
        required_evidence=["tracking_status"],
        risk_level=RiskLevel.HIGH,
    )


def test_parse_tool_schema_text():
    tools = parse_tool_schema_text(sample_tool_schema_text())

    assert len(tools) == 3
    assert tools[0]["name"] == "issue_shipping_refund"


def test_parse_invalid_tool_schema_returns_empty_list():
    tools = parse_tool_schema_text("not valid json")

    assert tools == []


def test_infer_expected_tool_name_for_shipping_refund():
    tools = parse_tool_schema_text(sample_tool_schema_text())
    rule = make_late_delivery_rule()

    expected_tool = infer_expected_tool_name(rule, tools)

    assert expected_tool == "issue_shipping_refund"


def test_infer_expected_tool_name_for_replacement():
    tools = parse_tool_schema_text(sample_tool_schema_text())
    rule = make_damaged_product_rule()

    expected_tool = infer_expected_tool_name(rule, tools)

    assert expected_tool == "create_replacement_order"


def test_infer_expected_tool_name_for_escalation():
    tools = parse_tool_schema_text(sample_tool_schema_text())
    rule = make_escalation_rule()

    expected_tool = infer_expected_tool_name(rule, tools)

    assert expected_tool == "escalate_to_human"


def test_build_tool_arguments_for_shipping_refund():
    args = build_tool_arguments(make_late_delivery_rule(), "issue_shipping_refund")

    assert args["order_id"] == "required_from_user_or_context"
    assert args["refund_type"] == "shipping_fee_only"
    assert args["eligibility_reason"] == "delivery_delay_gt_7_days"


def test_infer_missing_arguments():
    args = {
        "order_id": "required_from_user_or_context",
        "refund_type": "shipping_fee_only",
    }

    missing = infer_missing_arguments(args)

    assert missing == ["order_id"]


def test_generate_tool_use_cases():
    rule = make_late_delivery_rule()

    cases = generate_tool_use_cases(
        rules=[rule],
        tool_schema_text=sample_tool_schema_text(),
        project_id="support_demo",
        dataset_version="v0.1.0",
    )

    assert len(cases) == 1

    case = cases[0]

    assert case.test_type == TestType.TOOL_USE
    assert case.project_id == "support_demo"
    assert case.dataset_version == "v0.1.0"
    assert case.tool_expectation is not None
    assert case.tool_expectation.expected_tool == "issue_shipping_refund"
    assert case.tool_expectation.expected_tool_arguments["refund_type"] == "shipping_fee_only"
    assert case.tool_expectation.should_call_tool is False
    assert case.tool_expectation.should_ask_clarification_if_missing == ["order_id"]
    assert len(case.required_citations) == 1
    assert case.required_citations[0].chunk_id == "refund_policy_0001"
    assert "tool_use" in case.tags