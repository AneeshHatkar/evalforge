from __future__ import annotations

from typing import Dict, List, Optional

from src.schemas import Chunk, Document, EvalCase, KnowledgeRule


class InMemoryStore:
    """
    Temporary in-memory storage for Stage 2.

    Stage 3 will replace this with SQLite/PostgreSQL.
    """

    def __init__(self) -> None:
        self.projects: Dict[str, dict] = {}
        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []
        self.rules: List[KnowledgeRule] = []
        self.cases: List[EvalCase] = []
        self.quality_summary: Dict[str, object] = {}
        self.validation_results: Dict[str, object] = {}
        self.eval_results: List[dict] = []
        self.eval_summary: Optional[dict] = None

    def reset_pipeline(self) -> None:
        self.documents = []
        self.chunks = []
        self.rules = []
        self.cases = []
        self.quality_summary = {}
        self.validation_results = {}
        self.eval_results = []
        self.eval_summary = None


store = InMemoryStore()