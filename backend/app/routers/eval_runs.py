from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.api_schemas import EvalRunRequest, EvalRunResponse
from backend.app.state import store
from src.eval_runner import (
    demo_target_system,
    eval_results_to_dicts,
    eval_summary_to_dict,
    intentionally_bad_target_system,
    run_eval_dataset,
)

router = APIRouter(prefix="/eval-runs", tags=["eval-runs"])


@router.post("/demo", response_model=EvalRunResponse)
def run_demo_eval(request: EvalRunRequest) -> EvalRunResponse:
    if not store.cases:
        raise HTTPException(
            status_code=400,
            detail="No cases available. Run generation first.",
        )

    if request.target_system == "demo_target_system":
        target_fn = demo_target_system
    elif request.target_system == "intentionally_bad_target_system":
        target_fn = intentionally_bad_target_system
    else:
        raise HTTPException(
            status_code=400,
            detail="target_system must be demo_target_system or intentionally_bad_target_system",
        )

    results, summary = run_eval_dataset(
        store.cases,
        target_system=target_fn,
        target_system_name=request.target_system,
        dataset_version=store.cases[0].dataset_version if store.cases else "v0.1.0",
        pass_threshold=request.pass_threshold,
    )

    store.eval_results = eval_results_to_dicts(results)
    store.eval_summary = eval_summary_to_dict(summary)

    return EvalRunResponse(
        summary=store.eval_summary,
        result_count=len(store.eval_results),
    )


@router.get("/latest")
def get_latest_eval_run() -> dict:
    if store.eval_summary is None:
        raise HTTPException(
            status_code=404,
            detail="No eval run available.",
        )

    return {
        "summary": store.eval_summary,
        "results": store.eval_results,
    }