from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.schemas import (
    Citation,
    EvalCase,
    KnowledgeRule,
    RiskLevel,
    RuleType,
    TestType,
    ToolExpectation,
)


def parse_tool_schema_text(tool_schema_text: str) -> List[Dict[str, Any]]:
    """
    Parse a JSON tool schema document.

    Expected Stage 1 format:
    {
      "tools": [
        {
          "name": "issue_shipping_refund",
          "description": "...",
          "required_arguments": [...]
        }
      ]
    }
    """

    if not tool_schema_text or not tool_schema_text.strip():
        return []

    try:
        data = json.loads(tool_schema_text)
    except json.JSONDecodeError:
        return []

    tools = data.get("tools", [])

    if not isinstance(tools, list):
        return []

    valid_tools = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if "name" not in tool:
            continue

        valid_tools.append(tool)

    return valid_tools


def find_tool_by_name(tools: List[Dict[str, Any]], tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Find one tool from a parsed tool schema list.
    """

    for tool in tools:
        if tool.get("name") == tool_name:
            return tool

    return None


def infer_expected_tool_name(rule: KnowledgeRule, tools: List[Dict[str, Any]]) -> Optional[str]:
    """
    Infer which tool should be used based on the extracted rule text.

    Priority matters:
    - damaged/replacement cases should map to replacement tools even if the text also says refund
    - escalation cases should map to escalation tools
    - late-delivery shipping-fee refunds should map to shipping refund tools
    """

    text = rule.rule_text.lower()

    available_names = {tool.get("name") for tool in tools}

    # 1. Damaged/replacement cases should be checked before generic refund logic
    # because damaged-product rules often contain both "replacement" and "refund".
    if "damaged" in text or "replacement" in text or "photo evidence" in text:
        if "create_replacement_order" in available_names:
            return "create_replacement_order"

    # 2. Escalation cases should be explicit.
    if (
        "escalate" in text
        or "human review" in text
        or "lost-package investigation" in text
        or "repeated" in text
        or "duplicate" in text
        or "safety" in text
    ):
        if "escalate_to_human" in available_names:
            return "escalate_to_human"

    # 3. Shipping-fee refund cases.
    if (
        "shipping-fee refund" in text
        or "shipping fee refund" in text
        or ("refund" in text and ("late" in text or "delayed" in text or "delivery" in text))
    ):
        if "issue_shipping_refund" in available_names:
            return "issue_shipping_refund"

    return None


def build_tool_arguments(rule: KnowledgeRule, expected_tool: str) -> Dict[str, Any]:
    """
    Build expected tool arguments for a tool-use test case.

    Values are intentionally symbolic for Stage 1 because the benchmark checks
    whether the agent knows what arguments are required, not real order data.
    """

    if expected_tool == "issue_shipping_refund":
        return {
            "order_id": "required_from_user_or_context",
            "refund_type": "shipping_fee_only",
            "eligibility_reason": "delivery_delay_gt_7_days",
        }

    if expected_tool == "create_replacement_order":
        return {
            "order_id": "required_from_user_or_context",
            "damage_evidence": "required_from_user_or_context",
            "replacement_reason": "product_arrived_damaged",
        }

    if expected_tool == "escalate_to_human":
        return {
            "order_id": "required_from_user_or_context",
            "escalation_reason": infer_escalation_reason(rule),
        }

    return {}


def infer_escalation_reason(rule: KnowledgeRule) -> str:
    """
    Infer an escalation reason from the rule.
    """

    text = rule.rule_text.lower()

    if "lost" in text or "tracking" in text:
        return "lost_package_investigation"

    if "duplicate" in text or "repeated" in text or "already received compensation" in text:
        return "duplicate_or_repeated_refund_request"

    if "safety" in text:
        return "safety_concern"

    if "unclear" in text:
        return "unclear_case_requires_review"

    return "policy_requires_human_review"


def infer_missing_arguments(expected_tool_arguments: Dict[str, Any]) -> List[str]:
    """
    Any symbolic required_from_user_or_context argument should trigger clarification.
    """

    missing = []

    for key, value in expected_tool_arguments.items():
        if value == "required_from_user_or_context":
            missing.append(key)

    return missing


def make_tool_use_query(rule: KnowledgeRule, expected_tool: str) -> str:
    """
    Create a user query that should trigger tool-use behavior or clarification.
    """

    text = rule.rule_text.lower()

    if expected_tool == "issue_shipping_refund":
        return (
            "My delivery was 10 days late and I have not been compensated. "
            "Please process the shipping refund."
        )

    if expected_tool == "create_replacement_order":
        return (
            "My product arrived damaged and I can provide photo evidence. "
            "Please create a replacement order."
        )

    if expected_tool == "escalate_to_human":
        if "tracking" in text or "lost" in text:
            return (
                "My tracking has shown no movement for more than 10 days. "
                "Please handle this lost-package issue."
            )

        if "duplicate" in text or "repeated" in text:
            return (
                "I already received compensation but I am asking again for the same order. "
                "Can you process this?"
            )

        return "This case seems complicated and may need review. Please handle it."

    return "Please complete this support action for me."


def build_tool_expected_behavior(
    expected_tool: str,
    missing_arguments: List[str],
) -> str:
    """
    Explain what the target agent is expected to do.
    """

    if missing_arguments:
        readable = ", ".join(missing_arguments)
        return (
            f"The agent should identify that `{expected_tool}` is the relevant tool, "
            f"but should ask for missing required argument(s): {readable}, before calling it."
        )

    return (
        f"The agent should call `{expected_tool}` with the expected arguments and should not "
        "call unrelated tools."
    )


def build_tool_outline(
    expected_tool: str,
    expected_arguments: Dict[str, Any],
    missing_arguments: List[str],
) -> List[str]:
    """
    Build expected answer/tool behavior outline.
    """

    outline = [
        f"Relevant tool: {expected_tool}",
        "Do not call unrelated tools.",
        "Use only arguments supported by the user request and source policy.",
    ]

    if missing_arguments:
        outline.append(f"Ask for missing argument(s): {', '.join(missing_arguments)}")
        outline.append("Do not call the tool until required missing arguments are provided.")
    else:
        outline.append("Call the tool with the expected arguments.")

    for key, value in expected_arguments.items():
        outline.append(f"Expected argument `{key}`: {value}")

    return outline


def make_tool_citation(rule: KnowledgeRule) -> Citation:
    """
    Required citation showing which policy supports tool behavior.
    """

    return Citation(
        chunk_id=rule.source_chunk_id,
        source=rule.source,
        required_evidence=rule.rule_text.strip().rstrip("."),
    )


def should_generate_tool_case(rule: KnowledgeRule, tools: List[Dict[str, Any]]) -> bool:
    """
    Tool cases are generated only if a matching tool exists.
    """

    if not tools:
        return False

    return infer_expected_tool_name(rule, tools) is not None


def make_tool_test_id(rule: KnowledgeRule, index: int) -> str:
    """
    Create stable tool-use test IDs.

    Example:
    refund_policy_0001_tool_001
    """

    return f"{rule.source_chunk_id}_tool_{index:03d}"


def generate_tool_use_cases(
    rules: List[KnowledgeRule],
    tool_schema_text: str,
    project_id: str = "support_demo",
    dataset_version: str = "v0.1.0",
    max_cases: Optional[int] = None,
) -> List[EvalCase]:
    """
    Generate tool-use EvalCase objects from rules and a JSON tool schema.
    """

    tools = parse_tool_schema_text(tool_schema_text)
    cases: List[EvalCase] = []
    per_chunk_counts: Dict[str, int] = {}

    for rule in rules:
        if not should_generate_tool_case(rule, tools):
            continue

        if max_cases is not None and len(cases) >= max_cases:
            break

        expected_tool = infer_expected_tool_name(rule, tools)

        if expected_tool is None:
            continue

        expected_arguments = build_tool_arguments(rule, expected_tool)
        missing_arguments = infer_missing_arguments(expected_arguments)

        per_chunk_counts[rule.source_chunk_id] = per_chunk_counts.get(rule.source_chunk_id, 0) + 1
        case_index = per_chunk_counts[rule.source_chunk_id]

        tags = [
            "tool_use",
            "agent_workflow",
            expected_tool,
            rule.rule_type.value,
        ]

        if rule.section:
            tags.append(rule.section.lower().replace(" ", "_"))

        case = EvalCase(
            test_id=make_tool_test_id(rule, case_index),
            project_id=project_id,
            dataset_version=dataset_version,
            test_type=TestType.TOOL_USE,
            risk_level=RiskLevel.HIGH if expected_tool == "escalate_to_human" else rule.risk_level,
            user_query=make_tool_use_query(rule, expected_tool),
            expected_behavior=build_tool_expected_behavior(expected_tool, missing_arguments),
            expected_answer_outline=build_tool_outline(
                expected_tool=expected_tool,
                expected_arguments=expected_arguments,
                missing_arguments=missing_arguments,
            ),
            required_citations=[make_tool_citation(rule)],
            disallowed_behaviors=[
                "call an unrelated tool",
                "call the tool with missing required arguments",
                "invent argument values not provided by the user or context",
                "ignore policy conditions before taking action",
                "approve unsupported refunds, replacements, or escalations",
            ],
            tags=tags,
            tool_expectation=ToolExpectation(
                expected_tool=expected_tool,
                expected_tool_arguments=expected_arguments,
                should_call_tool=len(missing_arguments) == 0,
                should_ask_clarification_if_missing=missing_arguments,
            ),
            metadata={
                "source_rule_id": rule.rule_id,
                "source_chunk_id": rule.source_chunk_id,
                "generator": "tool_use_v1",
            },
        )

        cases.append(case)

    return cases