from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProjectCreateRequest(BaseModel):
    project_id: str = Field(default="support_demo")
    name: str = Field(default="Support Agent Benchmark")
    domain: str = Field(default="support_policy")
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    project_id: str
    name: str
    domain: str
    description: Optional[str] = None


class LoadSampleCorpusResponse(BaseModel):
    document_count: int
    chunk_count: int
    rule_count: int


class GenerationRunRequest(BaseModel):
    project_id: str = "support_demo"
    dataset_version: str = "v0.1.0"
    max_cases_per_type: int = Field(default=8, ge=1, le=50)
    citation_support_threshold: float = Field(default=0.35, ge=0.0, le=1.0)


class GenerationRunResponse(BaseModel):
    case_count: int
    quality_summary: Dict[str, object]


class EvalRunRequest(BaseModel):
    target_system: str = Field(default="demo_target_system")
    pass_threshold: float = Field(default=0.70, ge=0.0, le=1.0)


class EvalRunResponse(BaseModel):
    summary: Dict[str, object]
    result_count: int

class PipelineRunResponse(BaseModel):
    project_id: str
    dataset_version: str

    document_count: int
    chunk_count: int
    rule_count: int
    case_count: int

    quality_summary: Dict[str, object]

    eval_summary: Optional[Dict[str, object]] = None
    message: str = "Pipeline completed successfully."