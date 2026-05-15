from __future__ import annotations

from typing import Dict, List, Optional

from src.schemas import (
    Citation,
    EvalCase,
    KnowledgeRule,
    RiskLevel,
    RuleType,
    TestType,
)


def evidence_field_to_question(field: str) -> str:
    """
    Convert an internal evidence field into a natural clarification question.
    """

    mapping = {
        "order_id": "What is your order ID?",
        "delivery_date": "What date was the item delivered?",
        "promised_date": "What was the promised delivery date?",
        "delay_days": "How many days late was the delivery?",
        "compensation_status": "Have you already received compensation for this order?",
        "photo_evidence": "Can you provide photo evidence?",
        "purchase_date": "What was the purchase date?",
        "serial_number": "What is the product serial number?",
        "issue_description": "Can you describe the issue in more detail?",
        "tracking_status": "What does the tracking status currently show?",
    }

    return mapping.get(field, f"Can you provide the missing information for {field}?")


def make_ambiguous_query(rule: KnowledgeRule) -> str:
    """
    Create an underspecified user query from a rule.
    """

    text = rule.rule_text.lower()

    if "refund" in text and ("late" in text or "delayed" in text or "delivery" in text):
        return "Can I get a refund for my late package?"

    if "damaged" in text or "photo evidence" in text:
        return "My product arrived damaged. Can I get a replacement?"

    if "warranty" in text or "claim" in text:
        return "Can I start a warranty claim?"

    if "tracking" in text or "lost" in text:
        return "My package seems lost. What can you do?"

    if "already received compensation" in text or "duplicate" in text:
        return "Can I get another refund?"

    if "escalate" in text or "human review" in text:
        return "Can you escalate this?"

    if rule.section:
        return f"I need help with {rule.section.lower()}. What should I do?"

    return "Can you help me with this policy issue?"


def build_missing_information(rule: KnowledgeRule) -> List[str]:
    """
    Determine what information is missing from an ambiguous request.
    """

    if rule.required_evidence:
        return list(dict.fromkeys(rule.required_evidence))

    text = rule.rule_text.lower()
    missing: List[str] = []

    if "refund" in text:
        missing.append("order_id")

    if "delivery" in text or "late" in text or "delayed" in text:
        missing.extend(["delivery_date", "promised_date", "delay_days"])

    if "compensation" in text:
        missing.append("compensation_status")

    if "damaged" in text or "evidence" in text:
        missing.append("photo_evidence")

    if "warranty" in text:
        missing.extend(["order_id", "purchase_date", "serial_number", "issue_description"])

    if "tracking" in text:
        missing.append("tracking_status")

    return list(dict.fromkeys(missing))


def build_ambiguity_expected_behavior(rule: KnowledgeRule, missing_fields: List[str]) -> str:
    """
    Create expected behavior for ambiguity cases.
    """

    if missing_fields:
        readable_fields = ", ".join(missing_fields)
        return (
            "Do not make a final policy decision yet. Ask clarifying questions for the "
            f"missing information: {readable_fields}. After the missing details are provided, "
            "apply the source policy and cite the supporting source chunk."
        )

    return (
        "Do not guess missing facts. Ask a clarifying question before applying the policy. "
        "Use the source policy only after the user provides enough information."
    )


def build_ambiguity_outline(rule: KnowledgeRule, missing_fields: List[str]) -> List[str]:
    """
    Build expected answer outline for ambiguity test cases.
    """

    outline = [
        "Do not approve, deny, or complete the request immediately.",
        "State that more information is needed before applying the policy.",
    ]

    for field in missing_fields:
        outline.append(evidence_field_to_question(field))

    outline.append("Once the missing details are available, apply the policy using the cited source chunk.")

    return outline


def make_ambiguity_citation(rule: KnowledgeRule) -> Citation:
    """
    Create a citation explaining why clarification is needed.
    """

    evidence_text = rule.rule_text.strip().rstrip(".")

    return Citation(
        chunk_id=rule.source_chunk_id,
        source=rule.source,
        required_evidence=evidence_text,
    )


def should_generate_ambiguity_case(rule: KnowledgeRule) -> bool:
    """
    Only generate ambiguity cases when a rule depends on conditions or required evidence.
    """

    if rule.required_evidence:
        return True

    if rule.condition:
        return True

    if rule.rule_type in {RuleType.REQUIREMENT, RuleType.ELIGIBILITY, RuleType.TOOL_ACTION}:
        return True

    return False


def make_ambiguity_test_id(rule: KnowledgeRule, index: int) -> str:
    """
    Create stable ambiguity test IDs.

    Example:
    refund_policy_0001_ambiguity_001
    """

    return f"{rule.source_chunk_id}_ambiguity_{index:03d}"


def generate_ambiguity_cases(
    rules: List[KnowledgeRule],
    project_id: str = "support_demo",
    dataset_version: str = "v0.1.0",
    max_cases: Optional[int] = None,
) -> List[EvalCase]:
    """
    Generate ambiguity EvalCase objects from extracted rules.
    """

    cases: List[EvalCase] = []
    per_chunk_counts: Dict[str, int] = {}

    for rule in rules:
        if not should_generate_ambiguity_case(rule):
            continue

        if max_cases is not None and len(cases) >= max_cases:
            break

        missing_fields = build_missing_information(rule)

        per_chunk_counts[rule.source_chunk_id] = per_chunk_counts.get(rule.source_chunk_id, 0) + 1
        case_index = per_chunk_counts[rule.source_chunk_id]

        tags = [
            "ambiguity",
            "clarification_required",
            rule.rule_type.value,
        ]

        if rule.section:
            tags.append(rule.section.lower().replace(" ", "_"))

        case = EvalCase(
            test_id=make_ambiguity_test_id(rule, case_index),
            project_id=project_id,
            dataset_version=dataset_version,
            test_type=TestType.AMBIGUITY,
            risk_level=RiskLevel.MEDIUM if rule.risk_level == RiskLevel.LOW else rule.risk_level,
            user_query=make_ambiguous_query(rule),
            expected_behavior=build_ambiguity_expected_behavior(rule, missing_fields),
            expected_answer_outline=build_ambiguity_outline(rule, missing_fields),
            required_citations=[make_ambiguity_citation(rule)],
            disallowed_behaviors=[
                "approve the request without asking for missing information",
                "deny the request without checking required facts",
                "invent missing order, delivery, warranty, or compensation details",
                "ignore the policy conditions in the cited source chunk",
            ],
            tags=tags,
            metadata={
                "source_rule_id": rule.rule_id,
                "source_chunk_id": rule.source_chunk_id,
                "missing_fields": missing_fields,
                "generator": "ambiguity_v1",
            },
        )

        cases.append(case)

    return cases