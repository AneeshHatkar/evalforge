from __future__ import annotations

from fastapi import APIRouter

from backend.app.api_schemas import ProjectCreateRequest, ProjectResponse
from backend.app.state import store

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse)
def create_project(request: ProjectCreateRequest) -> ProjectResponse:
    project = {
        "project_id": request.project_id,
        "name": request.name,
        "domain": request.domain,
        "description": request.description,
    }

    store.projects[request.project_id] = project

    return ProjectResponse(**project)


@router.get("")
def list_projects() -> dict:
    return {
        "projects": list(store.projects.values())
    }