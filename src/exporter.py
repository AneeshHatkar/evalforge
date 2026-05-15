from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.schemas import EvalCase, ExportFormat, ReviewStatus


def ensure_output_directory(output_path: Union[str, Path]) -> Path:
    """
    Ensure the output directory exists.

    If output_path is a file path, create its parent directory.
    If output_path is a directory path, create the directory itself.
    """

    path = Path(output_path)

    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)

    return path


def case_to_serializable_dict(case: EvalCase) -> Dict[str, Any]:
    """
    Convert EvalCase to a JSON-serializable dictionary.

    Pydantic handles enums and datetime through mode='json'.
    """

    return case.model_dump(mode="json")


def cases_to_serializable_list(cases: List[EvalCase]) -> List[Dict[str, Any]]:
    """
    Convert a list of EvalCase objects into dictionaries.
    """

    return [case_to_serializable_dict(case) for case in cases]


def export_cases_json(
    cases: List[EvalCase],
    output_path: Union[str, Path],
    include_metadata: bool = True,
) -> Path:
    """
    Export cases as one JSON file.

    Format:
    {
      "metadata": {...},
      "cases": [...]
    }
    """

    output_path = ensure_output_directory(output_path)

    payload: Dict[str, Any] = {
        "cases": cases_to_serializable_list(cases)
    }

    if include_metadata:
        payload["metadata"] = {
            "exported_at": datetime.utcnow().isoformat(),
            "format": ExportFormat.JSON.value,
            "case_count": len(cases),
            "approved_case_count": count_cases_by_review_status(cases, ReviewStatus.APPROVED),
            "pending_case_count": count_cases_by_review_status(cases, ReviewStatus.PENDING),
            "rejected_case_count": count_cases_by_review_status(cases, ReviewStatus.REJECTED),
        }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_path


def export_cases_jsonl(
    cases: List[EvalCase],
    output_path: Union[str, Path],
) -> Path:
    """
    Export cases as JSONL.

    JSONL means one JSON object per line.
    This is useful for eval runners, CI jobs, and dataset pipelines.
    """

    output_path = ensure_output_directory(output_path)

    with output_path.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case_to_serializable_dict(case), ensure_ascii=False))
            file.write("\n")

    return output_path


def flatten_case_for_csv(case: EvalCase) -> Dict[str, Any]:
    """
    Flatten one EvalCase into a CSV-friendly row.

    Complex fields are converted into compact JSON strings.
    """

    citations = [citation.model_dump(mode="json") for citation in case.required_citations]

    tool_expectation = (
        case.tool_expectation.model_dump(mode="json")
        if case.tool_expectation is not None
        else None
    )

    return {
        "test_id": case.test_id,
        "project_id": case.project_id,
        "dataset_version": case.dataset_version,
        "test_type": case.test_type.value,
        "risk_level": case.risk_level.value,
        "review_status": case.review_status.value,
        "user_query": case.user_query,
        "expected_behavior": case.expected_behavior,
        "expected_answer_outline": json.dumps(case.expected_answer_outline, ensure_ascii=False),
        "required_citations": json.dumps(citations, ensure_ascii=False),
        "disallowed_behaviors": json.dumps(case.disallowed_behaviors, ensure_ascii=False),
        "tags": json.dumps(case.tags, ensure_ascii=False),
        "tool_expectation": json.dumps(tool_expectation, ensure_ascii=False),
        "validation_errors": json.dumps(case.validation_errors, ensure_ascii=False),
        "metadata": json.dumps(case.metadata, ensure_ascii=False),
        "created_at": case.created_at.isoformat(),
    }


def export_cases_csv(
    cases: List[EvalCase],
    output_path: Union[str, Path],
) -> Path:
    """
    Export cases as a flat CSV file.

    CSV is helpful for manual inspection in Excel, Google Sheets, or pandas.
    """

    output_path = ensure_output_directory(output_path)

    fieldnames = [
        "test_id",
        "project_id",
        "dataset_version",
        "test_type",
        "risk_level",
        "review_status",
        "user_query",
        "expected_behavior",
        "expected_answer_outline",
        "required_citations",
        "disallowed_behaviors",
        "tags",
        "tool_expectation",
        "validation_errors",
        "metadata",
        "created_at",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            writer.writerow(flatten_case_for_csv(case))

    return output_path


def export_quality_report(
    summary: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Export validation or benchmark quality summary as JSON.
    """

    output_path = ensure_output_directory(output_path)

    payload = {
        "metadata": {
            "exported_at": datetime.utcnow().isoformat(),
            "report_type": "evalforge_quality_report",
        },
        "summary": summary,
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_path


def count_cases_by_review_status(cases: List[EvalCase], status: ReviewStatus) -> int:
    """
    Count cases with a specific review status.
    """

    return sum(1 for case in cases if case.review_status == status)


def filter_cases_by_review_status(
    cases: List[EvalCase],
    statuses: List[ReviewStatus],
) -> List[EvalCase]:
    """
    Filter cases by review status.

    Example:
    export only approved cases.
    """

    allowed = set(statuses)
    return [case for case in cases if case.review_status in allowed]


def export_dataset_bundle(
    cases: List[EvalCase],
    output_dir: Union[str, Path],
    dataset_name: str = "support_eval_v1",
    quality_summary: Optional[Dict[str, Any]] = None,
    approved_only: bool = False,
) -> Dict[str, Path]:
    """
    Export a full Stage 1 dataset bundle.

    Creates:
    - JSON
    - JSONL
    - CSV
    - quality report if provided
    """

    output_dir = ensure_output_directory(output_dir)

    export_cases = cases

    if approved_only:
        export_cases = filter_cases_by_review_status(
            cases,
            statuses=[ReviewStatus.APPROVED],
        )

    paths: Dict[str, Path] = {}

    paths["json"] = export_cases_json(
        export_cases,
        output_dir / f"{dataset_name}.json",
    )

    paths["jsonl"] = export_cases_jsonl(
        export_cases,
        output_dir / f"{dataset_name}.jsonl",
    )

    paths["csv"] = export_cases_csv(
        export_cases,
        output_dir / f"{dataset_name}.csv",
    )

    if quality_summary is not None:
        paths["quality_report"] = export_quality_report(
            quality_summary,
            output_dir / f"{dataset_name}_quality_report.json",
        )

    return paths