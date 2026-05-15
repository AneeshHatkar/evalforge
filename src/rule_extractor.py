from __future__ import annotations

import re
from typing import List, Optional

from src.schemas import Chunk, KnowledgeRule, RiskLevel, RuleType


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

ELIGIBILITY_PATTERNS = [
    r"\beligible\b",
    r"\bmay request\b",
    r"\bmay receive\b",
    r"\bmay be eligible\b",
    r"\bcan receive\b",
    r"\bcan get\b",
]

REQUIREMENT_PATTERNS = [
    r"\bmust\b",
    r"\brequired\b",
    r"\brequires\b",
    r"\bneed to\b",
    r"\bneeds to\b",
    r"\bprovide\b",
]

PROHIBITION_PATTERNS = [
    r"\bnot eligible\b",
    r"\bdoes not cover\b",
    r"\bdo not\b",
    r"\bshould not\b",
    r"\bcannot\b",
    r"\bnot allowed\b",
]

ESCALATION_PATTERNS = [
    r"\bescalate\b",
    r"\bescalated\b",
    r"\bhuman review\b",
    r"\bhuman support\b",
    r"\binvestigation\b",
]

TOOL_ACTION_PATTERNS = [
    r"\brefund\b",
    r"\breplacement\b",
    r"\bclaim\b",
    r"\bprocess\b",
    r"\bissue\b",
]


def split_into_sentences(text: str) -> List[str]:
    """
    Split chunk text into policy-like sentences.

    This is intentionally simple for Stage 1.
    """

    text = text.strip()

    if not text:
        return []

    sentences = SENTENCE_SPLIT_PATTERN.split(text)
    cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]

    return cleaned


def pattern_matches(text: str, patterns: List[str]) -> bool:
    """
    Return True if any regex pattern appears in the text.
    """

    lowered = text.lower()

    return any(re.search(pattern, lowered) for pattern in patterns)


def infer_rule_type(sentence: str) -> RuleType:
    """
    Infer a rule type using deterministic keyword patterns.
    """

    lowered = sentence.lower()

    # Order matters. Prohibitions should be caught before eligibility
    # because "not eligible" contains "eligible".
    if pattern_matches(lowered, PROHIBITION_PATTERNS):
        return RuleType.PROHIBITION

    if pattern_matches(lowered, ESCALATION_PATTERNS):
        return RuleType.ESCALATION

    if pattern_matches(lowered, ELIGIBILITY_PATTERNS):
        return RuleType.ELIGIBILITY

    if pattern_matches(lowered, REQUIREMENT_PATTERNS):
        return RuleType.REQUIREMENT

    if pattern_matches(lowered, TOOL_ACTION_PATTERNS):
        return RuleType.TOOL_ACTION

    return RuleType.GENERAL_POLICY


def infer_risk_level(sentence: str, rule_type: RuleType) -> RiskLevel:
    """
    Assign a rough risk level for Stage 1.

    High-risk examples:
    - escalation
    - duplicate/refund abuse
    - safety concerns
    - unauthorized full refunds
    """

    lowered = sentence.lower()

    high_risk_terms = [
        "safety",
        "fraud",
        "unauthorized",
        "duplicate",
        "repeated",
        "human review",
        "escalate",
        "escalated",
        "lost-package investigation",
    ]

    medium_risk_terms = [
        "refund",
        "replacement",
        "warranty",
        "compensation",
        "damaged",
        "claim",
        "late",
        "delayed",
    ]

    if rule_type == RuleType.ESCALATION:
        return RiskLevel.HIGH

    if any(term in lowered for term in high_risk_terms):
        return RiskLevel.HIGH

    if any(term in lowered for term in medium_risk_terms):
        return RiskLevel.MEDIUM

    return RiskLevel.LOW


def extract_condition(sentence: str) -> Optional[str]:
    """
    Extract the condition part from policy sentences.

    Example:
    'Customers are eligible for a refund if delivery is delayed by more than 7 days.'
    condition -> 'delivery is delayed by more than 7 days'
    """

    patterns = [
        r"\bif\s+(.+)",
        r"\bwhen\s+(.+)",
        r"\bunless\s+(.+)",
        r"\bprovided that\s+(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if match:
            condition = match.group(1).strip()
            condition = condition.rstrip(".")
            return condition

    return None


def extract_expected_action(sentence: str) -> Optional[str]:
    """
    Extract a rough expected action from a sentence.

    For MVP, this is lightweight and readable rather than perfect.
    """

    condition_markers = [" if ", " when ", " unless ", " provided that "]

    lowered = sentence.lower()
    split_index = None

    for marker in condition_markers:
        marker_index = lowered.find(marker)
        if marker_index != -1:
            split_index = marker_index
            break

    if split_index is not None:
        action = sentence[:split_index].strip()
    else:
        action = sentence.strip()

    action = action.rstrip(".")

    return action or None


def infer_required_evidence(sentence: str) -> List[str]:
    """
    Infer fields or evidence the agent may need to answer correctly.

    This helps later with ambiguity and tool-use tests.
    """

    lowered = sentence.lower()
    evidence = []

    evidence_rules = {
        "order_id": ["order id", "order number"],
        "delivery_date": ["delivery date", "arrived", "delivery"],
        "promised_date": ["promised delivery", "promised date"],
        "delay_days": ["delayed", "late", "more than 7 days", "within 7 days"],
        "compensation_status": ["compensation", "already received"],
        "photo_evidence": ["photo evidence", "evidence", "upload evidence"],
        "purchase_date": ["purchase date"],
        "serial_number": ["serial number"],
        "issue_description": ["description of the issue", "description"],
        "tracking_status": ["tracking", "no movement"],
    }

    for evidence_name, keywords in evidence_rules.items():
        if any(keyword in lowered for keyword in keywords):
            evidence.append(evidence_name)

    return evidence


def should_extract_sentence(sentence: str) -> bool:
    """
    Decide whether a sentence is meaningful enough to become a rule.
    """

    if len(sentence.split()) < 5:
        return False

    rule_type = infer_rule_type(sentence)

    if rule_type != RuleType.GENERAL_POLICY:
        return True

    # Keep some general policy sentences if they contain useful thresholds.
    if re.search(r"\b\d+\b", sentence):
        return True

    return False


def make_rule_id(chunk_id: str, rule_number: int) -> str:
    """
    Create stable rule IDs.

    Example:
    refund_policy_0001 + 1 -> refund_policy_0001_rule_001
    """

    return f"{chunk_id}_rule_{rule_number:03d}"


def extract_rules_from_chunk(chunk: Chunk) -> List[KnowledgeRule]:
    """
    Extract structured rules from one chunk.
    """

    sentences = split_into_sentences(chunk.text)
    rules: List[KnowledgeRule] = []
    rule_counter = 1

    for sentence in sentences:
        if not should_extract_sentence(sentence):
            continue

        rule_type = infer_rule_type(sentence)
        condition = extract_condition(sentence)
        expected_action = extract_expected_action(sentence)
        risk_level = infer_risk_level(sentence, rule_type)
        required_evidence = infer_required_evidence(sentence)

        rules.append(
            KnowledgeRule(
                rule_id=make_rule_id(chunk.chunk_id, rule_counter),
                source_chunk_id=chunk.chunk_id,
                source=chunk.source,
                section=chunk.section,
                rule_type=rule_type,
                rule_text=sentence,
                condition=condition,
                expected_action=expected_action,
                required_evidence=required_evidence,
                risk_level=risk_level,
                confidence=0.75,
                metadata={
                    "document_id": chunk.document_id,
                    "chunk_char_count": len(chunk.text),
                },
            )
        )

        rule_counter += 1

    return rules


def extract_rules_from_chunks(chunks: List[Chunk]) -> List[KnowledgeRule]:
    """
    Extract rules from multiple chunks.
    """

    all_rules: List[KnowledgeRule] = []

    for chunk in chunks:
        all_rules.extend(extract_rules_from_chunk(chunk))

    return all_rules