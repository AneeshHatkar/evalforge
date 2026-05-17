from __future__ import annotations

from fastapi import FastAPI
from backend.app.db.session import init_db

from backend.app.config import settings
from backend.app.routers import (
    cases,
    corpora,
    eval_runs,
    exports,
    generation,
    health,
    pipeline,
    projects,
)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "EvalForge API: backend service for generating and running evaluation "
        "benchmarks for RAG systems and AI-agent workflows."
    ),
)
@app.on_event("startup")
def startup_event() -> None:
    init_db()

app.include_router(health.router)
app.include_router(projects.router)
app.include_router(corpora.router)
app.include_router(generation.router)
app.include_router(cases.router)
app.include_router(exports.router)
app.include_router(eval_runs.router)
app.include_router(pipeline.router)


@app.get("/")
def root() -> dict:
    return {
        "message": "EvalForge API is running.",
        "docs": "/docs",
        "health": "/health",
    }