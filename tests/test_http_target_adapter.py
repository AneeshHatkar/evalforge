import requests

from backend.app.api_schemas import HttpTargetConfig
from backend.app.services.http_target_adapter import (
    build_http_target_system,
    normalize_citations,
    normalize_tool_calls,
)
from src.schemas import Citation, EvalCase, RiskLevel, TestType


def make_case() -> EvalCase:
    return EvalCase(
        test_id="case_001",
        test_type=TestType.GROUNDED_QA,
        risk_level=RiskLevel.MEDIUM,
        user_query="Can I get a refund?",
        expected_behavior="Answer using policy.",
        expected_answer_outline=["Customer may be eligible for refund."],
        required_citations=[
            Citation(
                chunk_id="refund_policy_0001",
                source="refund_policy.md",
                required_evidence="Refund eligibility policy.",
            )
        ],
    )


def test_normalize_citations_from_strings():
    assert normalize_citations(["a", "b"]) == ["a", "b"]


def test_normalize_citations_from_dicts():
    raw = [{"chunk_id": "chunk_a"}, {"source_id": "chunk_b"}, {"id": "chunk_c"}]
    assert normalize_citations(raw) == ["chunk_a", "chunk_b", "chunk_c"]


def test_normalize_tool_calls():
    raw = [
        {
            "tool_name": "issue_shipping_refund",
            "args": {"order_id": "123"},
        }
    ]

    normalized = normalize_tool_calls(raw)

    assert normalized == [
        {
            "name": "issue_shipping_refund",
            "arguments": {"order_id": "123"},
        }
    ]


def test_http_target_adapter_builds_response(monkeypatch):
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "answer": "Customer may be eligible for refund.",
                "citations": ["refund_policy_0001"],
                "tool_calls": [],
            }

    def fake_post(url, json, timeout):
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    config = HttpTargetConfig(
        target_url="http://fake-target/answer",
        request_field="question",
        response_answer_field="answer",
        response_citations_field="citations",
        response_tool_calls_field="tool_calls",
    )

    target = build_http_target_system(config)
    response = target("Can I get a refund?", make_case())

    assert response.answer == "Customer may be eligible for refund."
    assert response.citations == ["refund_policy_0001"]
    assert response.tool_calls == []
    assert response.metadata["adapter"] == "http_target_adapter"