from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.app.db.models import (
    CaseRecord,
    ChunkRecord,
    DocumentRecord,
    PipelineRunRecord,
    ProjectRecord,
    RuleRecord,
)
from src.schemas import Chunk, Document, EvalCase, KnowledgeRule


def dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def loads(data: Optional[str]) -> Optional[dict]:
    if data is None:
        return None

    return json.loads(data)


def ensure_project(
    db: Session,
    project_id: str,
    name: Optional[str] = None,
    domain: str = "unknown",
    description: Optional[str] = None,
) -> ProjectRecord:
    project = (
        db.query(ProjectRecord)
        .filter(ProjectRecord.project_id == project_id)
        .first()
    )

    if project is not None:
        return project

    project = ProjectRecord(
        project_id=project_id,
        name=name or project_id,
        domain=domain,
        description=description,
    )

    db.add(project)
    db.commit()
    db.refresh(project)

    return project


def create_pipeline_run(
    db: Session,
    project_id: str,
    dataset_version: str,
    documents: List[Document],
    chunks: List[Chunk],
    rules: List[KnowledgeRule],
    cases: List[EvalCase],
    quality_summary: Dict[str, object],
    eval_summary: Optional[Dict[str, object]] = None,
) -> PipelineRunRecord:
    ensure_project(
        db=db,
        project_id=project_id,
        name=project_id,
        domain="auto_pipeline",
    )

    run_id = f"pipeline_{uuid.uuid4().hex[:12]}"

    run = PipelineRunRecord(
        run_id=run_id,
        project_id=project_id,
        dataset_version=dataset_version,
        document_count=len(documents),
        chunk_count=len(chunks),
        rule_count=len(rules),
        case_count=len(cases),
        quality_summary_json=dumps(quality_summary),
        eval_summary_json=dumps(eval_summary) if eval_summary is not None else None,
    )

    db.add(run)
    db.flush()

    for document in documents:
        db.add(
            DocumentRecord(
                pipeline_run_id=run_id,
                source_id=document.source_id,
                filename=document.filename,
                source_type=document.source_type.value,
                text=document.text,
                metadata_json=dumps(document.metadata),
            )
        )

    for chunk in chunks:
        db.add(
            ChunkRecord(
                pipeline_run_id=run_id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source=chunk.source,
                section=chunk.section,
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata_json=dumps(chunk.metadata),
            )
        )

    for rule in rules:
        db.add(
            RuleRecord(
                pipeline_run_id=run_id,
                rule_id=rule.rule_id,
                source_chunk_id=rule.source_chunk_id,
                source=rule.source,
                section=rule.section,
                rule_type=rule.rule_type.value,
                rule_text=rule.rule_text,
                condition=rule.condition,
                expected_action=rule.expected_action,
                risk_level=rule.risk_level.value,
                confidence=rule.confidence,
                payload_json=dumps(rule.model_dump(mode="json")),
            )
        )

    for case in cases:
        db.add(
            CaseRecord(
                pipeline_run_id=run_id,
                test_id=case.test_id,
                test_type=case.test_type.value,
                risk_level=case.risk_level.value,
                review_status=case.review_status.value,
                user_query=case.user_query,
                payload_json=dumps(case.model_dump(mode="json")),
            )
        )

    db.commit()
    db.refresh(run)

    return run


def list_pipeline_runs(
    db: Session,
    project_id: Optional[str] = None,
    limit: int = 20,
) -> List[PipelineRunRecord]:
    query = db.query(PipelineRunRecord)

    if project_id:
        query = query.filter(PipelineRunRecord.project_id == project_id)

    return (
        query.order_by(desc(PipelineRunRecord.created_at))
        .limit(limit)
        .all()
    )


def get_pipeline_run(
    db: Session,
    run_id: str,
) -> Optional[PipelineRunRecord]:
    return (
        db.query(PipelineRunRecord)
        .filter(PipelineRunRecord.run_id == run_id)
        .first()
    )


def pipeline_run_to_dict(run: PipelineRunRecord) -> dict:
    return {
        "run_id": run.run_id,
        "project_id": run.project_id,
        "dataset_version": run.dataset_version,
        "document_count": run.document_count,
        "chunk_count": run.chunk_count,
        "rule_count": run.rule_count,
        "case_count": run.case_count,
        "quality_summary": loads(run.quality_summary_json),
        "eval_summary": loads(run.eval_summary_json),
        "created_at": run.created_at.isoformat(),
    }


def get_cases_for_run(db: Session, run_id: str) -> List[dict]:
    rows = (
        db.query(CaseRecord)
        .filter(CaseRecord.pipeline_run_id == run_id)
        .all()
    )

    return [json.loads(row.payload_json) for row in rows]

def safe_float(value, default: float = 0.0) -> float:
    """
    Convert a value into float safely.
    """

    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def compare_numeric_metric(
    current_summary: dict,
    baseline_summary: dict,
    metric_name: str,
) -> dict:
    """
    Compare one numeric metric between two summaries.
    """

    current_value = safe_float(current_summary.get(metric_name))
    baseline_value = safe_float(baseline_summary.get(metric_name))
    delta = round(current_value - baseline_value, 4)

    return {
        "current": current_value,
        "baseline": baseline_value,
        "delta": delta,
    }


def compare_nested_scores(
    current_scores: dict,
    baseline_scores: dict,
) -> dict:
    """
    Compare nested score dictionaries, such as score_by_test_type.
    """

    all_keys = sorted(set(current_scores.keys()) | set(baseline_scores.keys()))
    comparison = {}

    for key in all_keys:
        current_value = safe_float(current_scores.get(key))
        baseline_value = safe_float(baseline_scores.get(key))
        comparison[key] = {
            "current": current_value,
            "baseline": baseline_value,
            "delta": round(current_value - baseline_value, 4),
        }

    return comparison


def compare_pipeline_runs(
    current_run: PipelineRunRecord,
    baseline_run: PipelineRunRecord,
    regression_threshold: float = -0.02,
) -> dict:
    """
    Compare two persisted pipeline runs.

    current_run is the run being evaluated.
    baseline_run is the previous/reference run.
    """

    current_quality = loads(current_run.quality_summary_json) or {}
    baseline_quality = loads(baseline_run.quality_summary_json) or {}

    current_eval = loads(current_run.eval_summary_json) or {}
    baseline_eval = loads(baseline_run.eval_summary_json) or {}

    metric_comparison = {
        "document_count": {
            "current": current_run.document_count,
            "baseline": baseline_run.document_count,
            "delta": current_run.document_count - baseline_run.document_count,
        },
        "chunk_count": {
            "current": current_run.chunk_count,
            "baseline": baseline_run.chunk_count,
            "delta": current_run.chunk_count - baseline_run.chunk_count,
        },
        "rule_count": {
            "current": current_run.rule_count,
            "baseline": baseline_run.rule_count,
            "delta": current_run.rule_count - baseline_run.rule_count,
        },
        "case_count": {
            "current": current_run.case_count,
            "baseline": baseline_run.case_count,
            "delta": current_run.case_count - baseline_run.case_count,
        },
        "validity_rate": compare_numeric_metric(
            current_quality,
            baseline_quality,
            "validity_rate",
        ),
        "citation_coverage": compare_numeric_metric(
            current_quality,
            baseline_quality,
            "citation_coverage",
        ),
        "total_errors": compare_numeric_metric(
            current_quality,
            baseline_quality,
            "total_errors",
        ),
        "total_warnings": compare_numeric_metric(
            current_quality,
            baseline_quality,
            "total_warnings",
        ),
        "average_score": compare_numeric_metric(
            current_eval,
            baseline_eval,
            "average_score",
        ),
        "pass_rate": compare_numeric_metric(
            current_eval,
            baseline_eval,
            "pass_rate",
        ),
    }

    current_type_scores = current_eval.get("score_by_test_type", {}) or {}
    baseline_type_scores = baseline_eval.get("score_by_test_type", {}) or {}

    score_by_test_type_comparison = compare_nested_scores(
        current_type_scores,
        baseline_type_scores,
    )

    regression_reasons = []

    for metric_name in ["validity_rate", "citation_coverage", "average_score", "pass_rate"]:
        delta = metric_comparison[metric_name]["delta"]

        if delta < regression_threshold:
            regression_reasons.append(
                f"{metric_name} decreased by {delta}"
            )

    for test_type, comparison in score_by_test_type_comparison.items():
        delta = comparison["delta"]

        if delta < regression_threshold:
            regression_reasons.append(
                f"{test_type} score decreased by {delta}"
            )

    # More errors is also a regression.
    if metric_comparison["total_errors"]["delta"] > 0:
        regression_reasons.append(
            f"total_errors increased by {metric_comparison['total_errors']['delta']}"
        )

    return {
        "current_run_id": current_run.run_id,
        "baseline_run_id": baseline_run.run_id,
        "current_project_id": current_run.project_id,
        "baseline_project_id": baseline_run.project_id,
        "current_dataset_version": current_run.dataset_version,
        "baseline_dataset_version": baseline_run.dataset_version,
        "metric_comparison": metric_comparison,
        "score_by_test_type_comparison": score_by_test_type_comparison,
        "regression_threshold": regression_threshold,
        "regression_detected": len(regression_reasons) > 0,
        "regression_reasons": regression_reasons,
    }

def get_quality_summary_for_run(db: Session, run_id: str) -> Optional[dict]:
    """
    Return persisted quality summary for one pipeline run.
    """

    run = get_pipeline_run(db=db, run_id=run_id)

    if run is None:
        return None

    return loads(run.quality_summary_json)


def get_eval_summary_for_run(db: Session, run_id: str) -> Optional[dict]:
    """
    Return persisted eval summary for one pipeline run, if available.
    """

    run = get_pipeline_run(db=db, run_id=run_id)

    if run is None:
        return None

    return loads(run.eval_summary_json)