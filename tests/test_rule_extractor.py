from src.rule_extractor import (
    extract_condition,
    extract_expected_action,
    extract_rules_from_chunk,
    infer_required_evidence,
    infer_risk_level,
    infer_rule_type,
    split_into_sentences,
)
from src.schemas import Chunk, RiskLevel, RuleType


def test_split_into_sentences():
    text = "Customers are eligible for refunds. Agents should escalate repeated requests."
    sentences = split_into_sentences(text)

    assert len(sentences) == 2
    assert sentences[0] == "Customers are eligible for refunds."
    assert sentences[1] == "Agents should escalate repeated requests."


def test_infer_rule_type_eligibility():
    sentence = "Customers are eligible for a shipping-fee refund if delivery is delayed."
    assert infer_rule_type(sentence) == RuleType.ELIGIBILITY


def test_infer_rule_type_prohibition_has_priority():
    sentence = "Customers are not eligible for another refund."
    assert infer_rule_type(sentence) == RuleType.PROHIBITION


def test_infer_rule_type_escalation():
    sentence = "Repeated refund requests should be escalated to human review."
    assert infer_rule_type(sentence) == RuleType.ESCALATION


def test_extract_condition_from_if_sentence():
    sentence = "Customers are eligible for a refund if delivery is delayed by more than 7 days."
    condition = extract_condition(sentence)

    assert condition == "delivery is delayed by more than 7 days"


def test_extract_expected_action():
    sentence = "Customers are eligible for a refund if delivery is delayed by more than 7 days."
    action = extract_expected_action(sentence)

    assert action == "Customers are eligible for a refund"


def test_infer_required_evidence():
    sentence = (
        "Customers are eligible for a shipping-fee refund if delivery is delayed "
        "by more than 7 days and the customer has not already received compensation."
    )

    evidence = infer_required_evidence(sentence)

    assert "delay_days" in evidence
    assert "compensation_status" in evidence


def test_infer_risk_level_for_escalation():
    sentence = "Repeated refund requests should be escalated to human review."
    risk = infer_risk_level(sentence, RuleType.ESCALATION)

    assert risk == RiskLevel.HIGH


def test_extract_rules_from_chunk():
    chunk = Chunk(
        chunk_id="refund_policy_0001",
        document_id="refund_policy",
        source="refund_policy.md",
        section="Late Delivery Refunds",
        text=(
            "Customers are eligible for a shipping-fee refund if delivery is delayed "
            "by more than 7 days and the customer has not already received compensation. "
            "Customers are not eligible for a full product refund only because the package arrived late."
        ),
    )

    rules = extract_rules_from_chunk(chunk)

    assert len(rules) == 2
    assert rules[0].rule_id == "refund_policy_0001_rule_001"
    assert rules[0].rule_type == RuleType.ELIGIBILITY
    assert rules[0].source_chunk_id == "refund_policy_0001"
    assert rules[1].rule_type == RuleType.PROHIBITION