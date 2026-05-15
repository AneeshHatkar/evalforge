from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.state import store
from src.exporter import cases_to_serializable_list

router = APIRouter(prefix="/exports", tags=["exports"])


@router.get("/json")
def export_json() -> dict:
    if not store.cases:
        raise HTTPException(
            status_code=400,
            detail="No cases available. Run generation first.",
        )

    return {
        "metadata": {
            "case_count": len(store.cases),
            "quality_summary": store.quality_summary,
        },
        "cases": cases_to_serializable_list(store.cases),
    }


@router.get("/quality-report")
def export_quality_report() -> dict:
    if not store.quality_summary:
        raise HTTPException(
            status_code=400,
            detail="No quality summary available. Run generation first.",
        )

    return {
        "summary": store.quality_summary
    }