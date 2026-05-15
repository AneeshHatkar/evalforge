from __future__ import annotations

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "EvalForge API"
    app_version: str = "0.2.0"
    environment: str = "local"
    default_project_id: str = "support_demo"
    default_dataset_version: str = "v0.1.0"


settings = Settings()