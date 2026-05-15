from src.citation_selector import (
    build_chunk_index,
    citation_exists,
    citation_is_supported,
    citation_support_score,
    ensure_case_has_citations,
    get_cited_chunks,
    select_best_citation_for_text,
    validate_case_citations,
)
from src.schemas import Chunk, Citation, EvalCase, RiskLevel, TestType


def make_refund_chunk() -> Chunk:
    return Chunk(
        chunk_id="refund_policy_0001",
        document_id="refund_policy",
        source="refund_policy.md",
        section="Late Delivery Refunds",
        text=(
            "Customers are eligible for a shipping-fee refund if delivery is delayed "
            "by more than 7 days and the customer has not already received compensation."
        ),
    )


def make_shipping_chunk() -> Chunk:
    return Chunk(
        chunk_id="shipping_policy_0001",
        document_id="shipping_policy",
        source="shipping_policy.md",
        section="Delayed Delivery",
        text=(
            "A delivery is considered late if it arrives more than 7 days after "
            "the promised delivery date."
        ),
    )


def make_valid_case() -> EvalCase:
    return EvalCase(
        test_id="refund_policy_0001_grounded_001",
        test_type=TestType.GROUNDED_QA,
        risk_level=RiskLevel.MEDIUM,
        user_query="Can I get a shipping refund if my package was more than 7 days late?",
        expected_behavior="Answer using the late-delivery refund policy.",
        expected_answer_outline=[
            "Customer may be eligible for a shipping-fee refund.",
            "Delivery must be delayed by more than 7 days.",
            "Customer must not already have received compensation.",
        ],
        required_citations=[
            Citation(
                chunk_id="refund_policy_0001",
                source="refund_policy.md",
                required_evidence=(
                    "Customers are eligible for a shipping-fee refund if delivery "
                    "is delayed by more than 7 days and the customer has not already "
                    "received compensation."
                ),
            )
        ],
    )


def test_build_chunk_index():
    chunks = [make_refund_chunk(), make_shipping_chunk()]

    index = build_chunk_index(chunks)

    assert "refund_policy_0001" in index
    assert "shipping_policy_0001" in index
    assert index["refund_policy_0001"].source == "refund_policy.md"


def test_citation_exists():
    chunk = make_refund_chunk()
    citation = Citation(
        chunk_id="refund_policy_0001",
        source="refund_policy.md",
        required_evidence="shipping-fee refund",
    )

    assert citation_exists(citation, build_chunk_index([chunk])) is True


def test_get_cited_chunks():
    chunk = make_refund_chunk()
    case = make_valid_case()

    cited_chunks = get_cited_chunks(case, [chunk])

    assert len(cited_chunks) == 1
    assert cited_chunks[0].chunk_id == "refund_policy_0001"


def test_citation_support_score_is_high_for_matching_text():
    chunk = make_refund_chunk()
    case = make_valid_case()
    citation = case.required_citations[0]

    score = citation_support_score(citation, chunk, case=case)

    assert score >= 0.7


def test_citation_is_supported():
    chunk = make_refund_chunk()
    case = make_valid_case()
    citation = case.required_citations[0]

    assert citation_is_supported(citation, chunk, case=case) is True


def test_validate_case_citations_valid_case():
    chunk = make_refund_chunk()
    case = make_valid_case()

    result = validate_case_citations(case, [chunk])

    assert result.is_valid is True
    assert result.errors == []


def test_validate_case_citations_missing_chunk():
    case = make_valid_case()

    result = validate_case_citations(case, [])

    assert result.is_valid is False
    assert any("missing chunk_id" in error for error in result.errors)


def test_select_best_citation_for_text():
    chunks = [make_refund_chunk(), make_shipping_chunk()]

    citations = select_best_citation_for_text(
        evidence_text="shipping-fee refund when delivery is delayed more than 7 days",
        chunks=chunks,
        top_k=1,
    )

    assert len(citations) == 1
    assert citations[0].chunk_id == "refund_policy_0001"


def test_ensure_case_has_citations():
    chunk = make_refund_chunk()

    case = EvalCase(
        test_id="refund_missing_citation_001",
        test_type=TestType.AMBIGUITY,
        risk_level=RiskLevel.MEDIUM,
        user_query="Can I get a refund?",
        expected_behavior="Ask for missing details before deciding eligibility.",
        expected_answer_outline=[
            "Ask whether delivery was delayed more than 7 days.",
            "Ask whether the customer already received compensation.",
        ],
        required_citations=[],
    )

    updated_case = ensure_case_has_citations(case, [chunk])

    assert len(updated_case.required_citations) == 1
    assert updated_case.required_citations[0].chunk_id == "refund_policy_0001"