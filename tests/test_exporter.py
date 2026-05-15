import json
from pathlib import Path

from src.exporter import (
    case_to_serializable_dict,
    count_cases_by_review_status,
    export_cases_csv,
    export_cases_json,
    export_cases_jsonl,
    export_dataset_bundle,
    export_quality_report,
    filter_cases_by_review_status,
    flatten_case_for_csv,
)
from src.schemas import Citation, EvalCase, ReviewStatus, RiskLevel, TestType


def make_case(
    test_id: str = "refund_policy_0001_grounded_001",
    review_status: ReviewStatus = ReviewStatus.PENDING,
) -> EvalCase:
    return EvalCase(
        test_id=test_id,
        test_type=TestType.GROUNDED_QA,
        risk_level=RiskLevel.MEDIUM,
        user_query="Can I get a shipping refund if my delivery was more than 7 days late?",
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
                required_evidence="Late delivery shipping-fee refund eligibility.",
            )
        ],
        tags=["grounded_answer", "refund"],
        review_status=review_status,
    )


def test_case_to_serializable_dict():
    case = make_case()

    data = case_to_serializable_dict(case)

    assert data["test_id"] == "refund_policy_0001_grounded_001"
    assert data["test_type"] == "grounded_policy_qa"
    assert data["risk_level"] == "medium"
    assert data["review_status"] == "pending_review"


def test_flatten_case_for_csv():
    case = make_case()

    row = flatten_case_for_csv(case)

    assert row["test_id"] == case.test_id
    assert row["test_type"] == "grounded_policy_qa"
    assert "shipping-fee refund" in row["expected_answer_outline"]
    assert "refund_policy_0001" in row["required_citations"]


def test_export_cases_json(tmp_path: Path):
    case = make_case()
    output_path = tmp_path / "dataset.json"

    exported_path = export_cases_json([case], output_path)

    assert exported_path.exists()

    data = json.loads(exported_path.read_text(encoding="utf-8"))

    assert data["metadata"]["case_count"] == 1
    assert len(data["cases"]) == 1
    assert data["cases"][0]["test_id"] == case.test_id


def test_export_cases_jsonl(tmp_path: Path):
    case_a = make_case("case_a")
    case_b = make_case("case_b")

    output_path = tmp_path / "dataset.jsonl"
    exported_path = export_cases_jsonl([case_a, case_b], output_path)

    lines = exported_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["test_id"] == "case_a"
    assert second["test_id"] == "case_b"


def test_export_cases_csv(tmp_path: Path):
    case = make_case()
    output_path = tmp_path / "dataset.csv"

    exported_path = export_cases_csv([case], output_path)

    content = exported_path.read_text(encoding="utf-8")

    assert "test_id" in content
    assert "refund_policy_0001_grounded_001" in content
    assert "grounded_policy_qa" in content


def test_export_quality_report(tmp_path: Path):
    summary = {
        "total_cases": 1,
        "valid_cases": 1,
        "validity_rate": 1.0,
    }

    output_path = tmp_path / "quality_report.json"
    exported_path = export_quality_report(summary, output_path)

    data = json.loads(exported_path.read_text(encoding="utf-8"))

    assert data["metadata"]["report_type"] == "evalforge_quality_report"
    assert data["summary"]["total_cases"] == 1


def test_count_cases_by_review_status():
    cases = [
        make_case("case_a", ReviewStatus.APPROVED),
        make_case("case_b", ReviewStatus.APPROVED),
        make_case("case_c", ReviewStatus.PENDING),
    ]

    assert count_cases_by_review_status(cases, ReviewStatus.APPROVED) == 2
    assert count_cases_by_review_status(cases, ReviewStatus.PENDING) == 1


def test_filter_cases_by_review_status():
    cases = [
        make_case("case_a", ReviewStatus.APPROVED),
        make_case("case_b", ReviewStatus.REJECTED),
        make_case("case_c", ReviewStatus.PENDING),
    ]

    approved = filter_cases_by_review_status(cases, [ReviewStatus.APPROVED])

    assert len(approved) == 1
    assert approved[0].test_id == "case_a"


def test_export_dataset_bundle(tmp_path: Path):
    case = make_case("case_a", ReviewStatus.APPROVED)

    paths = export_dataset_bundle(
        cases=[case],
        output_dir=tmp_path,
        dataset_name="test_dataset",
        quality_summary={"total_cases": 1},
    )

    assert paths["json"].exists()
    assert paths["jsonl"].exists()
    assert paths["csv"].exists()
    assert paths["quality_report"].exists()