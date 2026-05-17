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