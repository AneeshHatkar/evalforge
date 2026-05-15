from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.app.api_schemas import GenerationRunRequest, GenerationRunResponse
from backend.app.state import store
from src.generators.adversarial import generate_adversarial_cases
from src.generators.ambiguity import generate_ambiguity_cases
from src.generators.grounded_qa import generate_grounded_qa_cases
from src.generators.tool_use import generate_tool_use_cases
from src.validator import attach_validation_errors, summarize_validation_results, validate_cases

router = APIRouter(prefix="/generation", tags=["generation"])


def get_sample_tool_schema_text() -> str:
    path = Path("sample_docs/support_tool_schema.json")

    if path.exists():
        return path.read_text(encoding="utf-8")

    return ""


@router.post("/run", response_model=GenerationRunResponse)
def run_generation(request: GenerationRunRequest) -> GenerationRunResponse:
    if not store.rules:
        raise HTTPException(
            status_code=400,
            detail="No rules available. Call /corpora/load-sample first.",
        )

    tool_schema_text = get_sample_tool_schema_text()

    cases = []
    cases.extend(
        generate_grounded_qa_cases(
            store.rules,
            project_id=request.project_id,
            dataset_version=request.dataset_version,
            max_cases=request.max_cases_per_type,
        )
    )
    cases.extend(
        generate_ambiguity_cases(
            store.rules,
            project_id=request.project_id,
            dataset_version=request.dataset_version,
            max_cases=request.max_cases_per_type,
        )
    )
    cases.extend(
        generate_adversarial_cases(
            store.rules,
            project_id=request.project_id,
            dataset_version=request.dataset_version,
            max_cases=request.max_cases_per_type,
        )
    )

    if tool_schema_text:
        cases.extend(
            generate_tool_use_cases(
                store.rules,
                tool_schema_text=tool_schema_text,
                project_id=request.project_id,
                dataset_version=request.dataset_version,
                max_cases=request.max_cases_per_type,
            )
        )

    validation_results = validate_cases(
        cases,
        store.chunks,
        citation_support_threshold=request.citation_support_threshold,
    )
    cases_with_errors = attach_validation_errors(cases, validation_results)
    quality_summary = summarize_validation_results(cases_with_errors, validation_results)

    store.cases = cases_with_errors
    store.validation_results = validation_results
    store.quality_summary = quality_summary

    return GenerationRunResponse(
        case_count=len(cases_with_errors),
        quality_summary=quality_summary,
    )