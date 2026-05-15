from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.state import store
from src.schemas import ReviewStatus

router = APIRouter(prefix="/cases", tags=["cases"])


@router.get("")
def list_cases() -> dict:
    return {
        "case_count": len(store.cases),
        "cases": [case.model_dump(mode="json") for case in store.cases],
    }


@router.get("/{test_id}")
def get_case(test_id: str) -> dict:
    for case in store.cases:
        if case.test_id == test_id:
            return case.model_dump(mode="json")

    raise HTTPException(status_code=404, detail=f"Case not found: {test_id}")


@router.patch("/{test_id}/review")
def update_case_review_status(test_id: str, review_status: ReviewStatus) -> dict:
    updated_cases = []

    found = False
    updated_case = None

    for case in store.cases:
        if case.test_id == test_id:
            found = True
            case_data = case.model_dump()
            case_data["review_status"] = review_status
            updated_case = type(case)(**case_data)
            updated_cases.append(updated_case)
        else:
            updated_cases.append(case)

    if not found:
        raise HTTPException(status_code=404, detail=f"Case not found: {test_id}")

    store.cases = updated_cases

    return {
        "updated": True,
        "case": updated_case.model_dump(mode="json"),
    }