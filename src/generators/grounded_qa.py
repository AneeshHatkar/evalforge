from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.schemas import (
    Citation,
    EvalCase,
    KnowledgeRule,
    RiskLevel,
    RuleType,
    TestType,
)


def clean_text_for_outline(text: str) -> str:
    """
    Clean a rule sentence so it can be used as an expected answer bullet.
    """

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".")

    return text


def sentence_to_question(rule: KnowledgeRule) -> str:
    """
    Convert a structured policy rule into a natural user question.

    This is deterministic for Stage 1. Later, we can replace this with
    LLM-based question generation while keeping the same EvalCase schema.
    """

    rule_text = rule.rule_text.lower()
    condition = rule.condition or ""

    section = (rule.section or "").lower()

    if "late" in rule_text or "delayed" in rule_text or "delivery" in rule_text:
        if "refund" in rule_text:
            return (
                "My package arrived more than 7 days late and I have not already "
                "received compensation. Am I eligible for a shipping-fee refund?"
            )

        return (
            "My delivery seems delayed. What does the policy say I should expect?"
        )

    if "damaged" in rule_text or "photo evidence" in rule_text:
        return (
            "My product arrived damaged. What do I need to provide to request a refund "
            "or replacement?"
        )

    if "warranty" in rule_text or "claim" in rule_text:
        return (
            "I want to start a warranty claim. What information do I need to provide?"
        )

    if "tracking" in rule_text or "lost" in rule_text:
        return (
            "My tracking has not moved for more than 10 days. What should support do?"
        )

    if "duplicate" in rule_text or "already received compensation" in rule_text:
        return (
            "I already received compensation for this order, but I want another refund. "
            "Am I eligible?"
        )

    if "escalate" in rule_text or "human review" in rule_text:
        return (
            "This case seems complicated. When should it be escalated to human review?"
        )

    if condition:
        return f"What should happen if {condition}?"

    if section:
        return f"What does the {section} policy say?"

    return "What does the policy say about this situation?"


def build_expected_answer_outline(rule: KnowledgeRule) -> List[str]:
    """
    Build a readable expected answer outline from a KnowledgeRule.
    """

    outline: List[str] = []

    if rule.expected_action:
        outline.append(clean_text_for_outline(rule.expected_action))

    if rule.condition:
        outline.append(f"Condition: {clean_text_for_outline(rule.condition)}")

    if rule.required_evidence:
        readable_evidence = ", ".join(rule.required_evidence)
        outline.append(f"Relevant evidence fields: {readable_evidence}")

    if not outline:
        outline.append(clean_text_for_outline(rule.rule_text))

    return outline


def build_expected_behavior(rule: KnowledgeRule) -> str:
    """
    Build expected behavior for the target AI system.
    """

    if rule.rule_type == RuleType.PROHIBITION:
        return (
            "Apply the policy restriction accurately. Do not approve or suggest an "
            "action that the source policy prohibits. Cite the supporting source chunk."
        )

    if rule.rule_type == RuleType.ESCALATION:
        return (
            "Identify that the case requires escalation or human review according to "
            "the policy. Cite the supporting source chunk."
        )

    if rule.rule_type == RuleType.REQUIREMENT:
        return (
            "List the required information or evidence from the policy before the "
            "request can proceed. Cite the supporting source chunk."
        )

    if rule.rule_type == RuleType.ELIGIBILITY:
        return (
            "Answer the eligibility question using the source policy conditions. "
            "Mention all required conditions and cite the supporting source chunk."
        )

    return (
        "Answer using only the provided source policy. Include the relevant condition "
        "or rule and cite the supporting source chunk."
    )


def make_required_citation(rule: KnowledgeRule) -> Citation:
    """
    Convert a KnowledgeRule into a required citation.
    """

    evidence = clean_text_for_outline(rule.rule_text)

    return Citation(
        chunk_id=rule.source_chunk_id,
        source=rule.source,
        required_evidence=evidence,
    )


def should_generate_grounded_case(rule: KnowledgeRule) -> bool:
    """
    Decide whether a rule is suitable for grounded QA generation.
    """

    if rule.rule_type in {
        RuleType.ELIGIBILITY,
        RuleType.REQUIREMENT,
        RuleType.PROHIBITION,
        RuleType.ESCALATION,
        RuleType.TOOL_ACTION,
        RuleType.GENERAL_POLICY,
    }:
        return True

    return False


def make_grounded_test_id(rule: KnowledgeRule, index: int) -> str:
    """
    Create stable grounded QA test IDs.

    Example:
    refund_policy_0001_rule_001 -> refund_policy_0001_grounded_001
    """

    chunk_part = rule.source_chunk_id
    return f"{chunk_part}_grounded_{index:03d}"


def generate_grounded_qa_cases(
    rules: List[KnowledgeRule],
    project_id: str = "support_demo",
    dataset_version: str = "v0.1.0",
    max_cases: Optional[int] = None,
) -> List[EvalCase]:
    """
    Generate grounded QA EvalCase objects from extracted rules.
    """

    cases: List[EvalCase] = []
    per_chunk_counts: Dict[str, int] = {}

    for rule in rules:
        if not should_generate_grounded_case(rule):
            continue

        if max_cases is not None and len(cases) >= max_cases:
            break

        per_chunk_counts[rule.source_chunk_id] = per_chunk_counts.get(rule.source_chunk_id, 0) + 1
        case_index = per_chunk_counts[rule.source_chunk_id]

        tags = [
            "grounded_answer",
            rule.rule_type.value,
        ]

        if rule.section:
            tags.append(rule.section.lower().replace(" ", "_"))

        case = EvalCase(
            test_id=make_grounded_test_id(rule, case_index),
            project_id=project_id,
            dataset_version=dataset_version,
            test_type=TestType.GROUNDED_QA,
            risk_level=rule.risk_level if rule.risk_level else RiskLevel.LOW,
            user_query=sentence_to_question(rule),
            expected_behavior=build_expected_behavior(rule),
            expected_answer_outline=build_expected_answer_outline(rule),
            required_citations=[make_required_citation(rule)],
            disallowed_behaviors=[
                "answer without citing the required source chunk",
                "invent policy details not present in the source",
                "ignore explicit policy conditions or restrictions",
            ],
            tags=tags,
            metadata={
                "source_rule_id": rule.rule_id,
                "source_chunk_id": rule.source_chunk_id,
                "generator": "grounded_qa_v1",
            },
        )

        cases.append(case)

    return cases