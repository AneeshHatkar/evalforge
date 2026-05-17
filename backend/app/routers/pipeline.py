import tempfile
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.api_schemas import PipelineRunResponse
from backend.app.db.repositories import (
    create_pipeline_run,
    get_cases_for_run,
    get_pipeline_run,
    list_pipeline_runs,
    pipeline_run_to_dict,
)
from backend.app.db.session import get_db
from backend.app.state import store
from src.chunker import chunk_documents
from src.document_loader import load_documents
from src.eval_runner import (
    demo_target_system,
    eval_results_to_dicts,
    eval_summary_to_dict,
    intentionally_bad_target_system,
    run_eval_dataset,
)
from src.generators.adversarial import generate_adversarial_cases
from src.generators.ambiguity import generate_ambiguity_cases
from src.generators.grounded_qa import generate_grounded_qa_cases
from src.generators.tool_use import generate_tool_use_cases
from src.rule_extractor import extract_rules_from_chunks
from src.validator import attach_validation_errors, summarize_validation_results, validate_cases

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


SUPPORTED_UPLOAD_EXTENSIONS = {".md", ".txt", ".json", ".csv", ".pdf"}


async def save_uploads_to_temp_dir(files: List[UploadFile]) -> List[Path]:
    """
    Save uploaded files to a temporary directory so the existing document loader
    can process them as normal file paths.
    """

    temp_dir = Path(tempfile.mkdtemp(prefix="evalforge_api_uploads_"))
    saved_paths: List[Path] = []

    for uploaded_file in files:
        original_name = uploaded_file.filename or "uploaded_file"
        suffix = Path(original_name).suffix.lower()

        if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type: {suffix}. "
                    f"Supported types: {sorted(SUPPORTED_UPLOAD_EXTENSIONS)}"
                ),
            )

        safe_name = Path(original_name).name
        output_path = temp_dir / safe_name

        content = await uploaded_file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file is empty: {original_name}",
            )

        output_path.write_bytes(content)
        saved_paths.append(output_path)

    return saved_paths


def detect_tool_schema_text_from_saved_files(saved_paths: List[Path]) -> str:
    """
    Detect tool schema files from uploaded JSON files.

    Stage 2 uses a simple filename heuristic:
    - contains 'tool'
    - contains 'schema'
    """

    for path in saved_paths:
        filename = path.name.lower()

        if path.suffix.lower() == ".json" and (
            "tool" in filename or "schema" in filename
        ):
            return path.read_text(encoding="utf-8")

    return ""


def generate_cases_from_rules(
    project_id: str,
    dataset_version: str,
    max_cases_per_type: int,
    tool_schema_text: str,
):
    """
    Run all EvalForge Stage 2 generators.
    """

    cases = []

    cases.extend(
        generate_grounded_qa_cases(
            store.rules,
            project_id=project_id,
            dataset_version=dataset_version,
            max_cases=max_cases_per_type,
        )
    )

    cases.extend(
        generate_ambiguity_cases(
            store.rules,
            project_id=project_id,
            dataset_version=dataset_version,
            max_cases=max_cases_per_type,
        )
    )

    cases.extend(
        generate_adversarial_cases(
            store.rules,
            project_id=project_id,
            dataset_version=dataset_version,
            max_cases=max_cases_per_type,
        )
    )

    if tool_schema_text.strip():
        cases.extend(
            generate_tool_use_cases(
                store.rules,
                tool_schema_text=tool_schema_text,
                project_id=project_id,
                dataset_version=dataset_version,
                max_cases=max_cases_per_type,
            )
        )

    return cases


@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(
    files: Annotated[
        List[UploadFile],
        File(description="Upload one or more .md, .txt, .json, .csv, or .pdf files."),
    ],
    project_id: Annotated[str, Form()] = "support_demo",
    dataset_version: Annotated[str, Form()] = "v0.1.0",
    max_cases_per_type: Annotated[int, Form()] = 8,
    citation_support_threshold: Annotated[float, Form()] = 0.35,
    run_eval: Annotated[bool, Form()] = False,
    target_system: Annotated[str, Form()] = "demo_target_system",
    pass_threshold: Annotated[float, Form()] = 0.70,
    db: Session = Depends(get_db),
) -> PipelineRunResponse:
    """
    One-shot EvalForge pipeline.

    User uploads RAG docs, policy files, tool schemas, CSVs, or PDFs.
    EvalForge automatically loads, chunks, extracts rules, generates benchmark
    cases, validates them, optionally runs evaluation, and persists the run.
    """

    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one file.")

    if max_cases_per_type < 1 or max_cases_per_type > 50:
        raise HTTPException(
            status_code=400,
            detail="max_cases_per_type must be between 1 and 50.",
        )

    if citation_support_threshold < 0 or citation_support_threshold > 1:
        raise HTTPException(
            status_code=400,
            detail="citation_support_threshold must be between 0 and 1.",
        )

    if pass_threshold < 0 or pass_threshold > 1:
        raise HTTPException(
            status_code=400,
            detail="pass_threshold must be between 0 and 1.",
        )

    store.reset_pipeline()

    saved_paths = await save_uploads_to_temp_dir(files)
    tool_schema_text = detect_tool_schema_text_from_saved_files(saved_paths)

    documents = load_documents(saved_paths)
    chunks = chunk_documents(documents, max_chars=700, overlap_chars=100)
    rules = extract_rules_from_chunks(chunks)

    store.documents = documents
    store.chunks = chunks
    store.rules = rules

    cases = generate_cases_from_rules(
        project_id=project_id,
        dataset_version=dataset_version,
        max_cases_per_type=max_cases_per_type,
        tool_schema_text=tool_schema_text,
    )

    validation_results = validate_cases(
        cases,
        chunks,
        citation_support_threshold=citation_support_threshold,
    )

    cases_with_errors = attach_validation_errors(cases, validation_results)
    quality_summary = summarize_validation_results(cases_with_errors, validation_results)

    store.cases = cases_with_errors
    store.validation_results = validation_results
    store.quality_summary = quality_summary

    eval_summary: Optional[dict] = None

    if run_eval:
        if target_system == "demo_target_system":
            target_fn = demo_target_system
        elif target_system == "intentionally_bad_target_system":
            target_fn = intentionally_bad_target_system
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "target_system must be demo_target_system or "
                    "intentionally_bad_target_system"
                ),
            )

        eval_results, eval_run_summary = run_eval_dataset(
            cases_with_errors,
            target_system=target_fn,
            target_system_name=target_system,
            dataset_version=dataset_version,
            pass_threshold=pass_threshold,
        )

        store.eval_results = eval_results_to_dicts(eval_results)
        store.eval_summary = eval_summary_to_dict(eval_run_summary)
        eval_summary = store.eval_summary

    pipeline_run_record = create_pipeline_run(
        db=db,
        project_id=project_id,
        dataset_version=dataset_version,
        documents=documents,
        chunks=chunks,
        rules=rules,
        cases=cases_with_errors,
        quality_summary=quality_summary,
        eval_summary=eval_summary,
    )

    return PipelineRunResponse(
        project_id=project_id,
        dataset_version=dataset_version,
        pipeline_run_id=pipeline_run_record.run_id,
        document_count=len(documents),
        chunk_count=len(chunks),
        rule_count=len(rules),
        case_count=len(cases_with_errors),
        quality_summary=quality_summary,
        eval_summary=eval_summary,
        message=(
            "Pipeline completed successfully. Uploaded files were ingested, chunked, "
            "converted into rules, used to generate benchmark cases, validated, and "
            "stored in backend memory and SQLite persistence."
        ),
    )


@router.get("/runs")
def list_runs(
    project_id: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db),
) -> dict:
    """
    List persisted pipeline runs.

    Optional:
    - project_id: filter runs by project
    - limit: maximum number of runs to return
    """

    runs = list_pipeline_runs(db=db, project_id=project_id, limit=limit)

    return {
        "run_count": len(runs),
        "runs": [pipeline_run_to_dict(run) for run in runs],
    }


@router.get("/runs/{run_id}")
def get_run(
    run_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Get metadata and summaries for one persisted pipeline run.
    """

    run = get_pipeline_run(db=db, run_id=run_id)

    if run is None:
        raise HTTPException(status_code=404, detail=f"Pipeline run not found: {run_id}")

    return pipeline_run_to_dict(run)


@router.get("/runs/{run_id}/cases")
def get_run_cases(
    run_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Get generated cases for one persisted pipeline run.
    """

    run = get_pipeline_run(db=db, run_id=run_id)

    if run is None:
        raise HTTPException(status_code=404, detail=f"Pipeline run not found: {run_id}")

    cases = get_cases_for_run(db=db, run_id=run_id)

    return {
        "run_id": run_id,
        "case_count": len(cases),
        "cases": cases,
    }

from backend.app.db.repositories import (
    compare_pipeline_runs,
    create_pipeline_run,
    get_cases_for_run,
    get_pipeline_run,
    list_pipeline_runs,
    pipeline_run_to_dict,
)

@router.get("/runs")
def list_runs(
    project_id: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db),
) -> dict:
    runs = list_pipeline_runs(db=db, project_id=project_id, limit=limit)

    return {
        "run_count": len(runs),
        "runs": [pipeline_run_to_dict(run) for run in runs],
    }


@router.get("/runs/{run_id}")
def get_run(
    run_id: str,
    db: Session = Depends(get_db),
) -> dict:
    run = get_pipeline_run(db=db, run_id=run_id)

    if run is None:
        raise HTTPException(status_code=404, detail=f"Pipeline run not found: {run_id}")

    return pipeline_run_to_dict(run)


@router.get("/runs/{run_id}/cases")
def get_run_cases(
    run_id: str,
    db: Session = Depends(get_db),
) -> dict:
    run = get_pipeline_run(db=db, run_id=run_id)

    if run is None:
        raise HTTPException(status_code=404, detail=f"Pipeline run not found: {run_id}")

    cases = get_cases_for_run(db=db, run_id=run_id)

    return {
        "run_id": run_id,
        "case_count": len(cases),
        "cases": cases,
    }

@router.get("/runs/{run_id}/compare/{baseline_run_id}")
def compare_runs(
    run_id: str,
    baseline_run_id: str,
    regression_threshold: float = -0.02,
    db: Session = Depends(get_db),
) -> dict:
    """
    Compare two persisted pipeline runs.

    Use this to detect whether a newer run improved or regressed compared
    with a baseline run.
    """

    current_run = get_pipeline_run(db=db, run_id=run_id)

    if current_run is None:
        raise HTTPException(status_code=404, detail=f"Pipeline run not found: {run_id}")

    baseline_run = get_pipeline_run(db=db, run_id=baseline_run_id)

    if baseline_run is None:
        raise HTTPException(
            status_code=404,
            detail=f"Baseline pipeline run not found: {baseline_run_id}",
        )

    return compare_pipeline_runs(
        current_run=current_run,
        baseline_run=baseline_run,
        regression_threshold=regression_threshold,
    )