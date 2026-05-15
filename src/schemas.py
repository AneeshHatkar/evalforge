from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SourceType(str, Enum):
    MARKDOWN = "markdown"
    TEXT = "text"
    JSON = "json"
    PDF = "pdf"
    UNKNOWN = "unknown"


class TestType(str, Enum):
    GROUNDED_QA = "grounded_policy_qa"
    CITATION_ALIGNMENT = "citation_alignment"
    AMBIGUITY = "ambiguity"
    REFUSAL = "refusal"
    ADVERSARIAL = "adversarial"
    TOOL_USE = "tool_use_correctness"
    BOUNDARY_CONDITION = "boundary_condition"
    REGRESSION = "regression"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReviewStatus(str, Enum):
    PENDING = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_FIX = "needs_fix"

class RuleType(str, Enum):
    ELIGIBILITY = "eligibility"
    REQUIREMENT = "requirement"
    PROHIBITION = "prohibition"
    ESCALATION = "escalation"
    TOOL_ACTION = "tool_action"
    DEFINITION = "definition"
    GENERAL_POLICY = "general_policy"

class Document(BaseModel):
    """
    Represents a source document uploaded into EvalForge.

    Example:
    refund_policy.md
    shipping_policy.md
    support_tool_schema.json
    """

    source_id: str = Field(..., description="Stable ID for the document.")
    filename: str = Field(..., description="Original filename.")
    source_type: SourceType = Field(default=SourceType.UNKNOWN)
    text: str = Field(..., description="Extracted normalized text.")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("source_id", "filename")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Value cannot be empty.")
        return value.strip()

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Document text cannot be empty.")
        return value.strip()


class Chunk(BaseModel):
    """
    A citation-level piece of evidence.

    Every generated test case should point back to one or more chunks.
    """

    chunk_id: str = Field(..., description="Stable chunk ID.")
    document_id: str = Field(..., description="Document/source ID this chunk came from.")
    source: str = Field(..., description="Original filename.")
    section: Optional[str] = Field(default=None, description="Section heading if available.")
    text: str = Field(..., description="Chunk text.")
    start_char: Optional[int] = Field(default=None, ge=0)
    end_char: Optional[int] = Field(default=None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id", "document_id", "source")
    @classmethod
    def required_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Required string field cannot be empty.")
        return value.strip()

    @field_validator("text")
    @classmethod
    def chunk_text_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Chunk text cannot be empty.")
        return value.strip()

    @model_validator(mode="after")
    def validate_offsets(self) -> "Chunk":
        if self.start_char is not None and self.end_char is not None:
            if self.end_char < self.start_char:
                raise ValueError("end_char must be greater than or equal to start_char.")
        return self

class KnowledgeRule(BaseModel):
    """
    A structured rule, fact, procedure, or constraint extracted from a source chunk.

    Stage 1 uses deterministic extraction.
    Later versions can use LLM structured extraction.
    """

    rule_id: str = Field(..., description="Stable rule ID.")
    source_chunk_id: str = Field(..., description="Chunk where the rule came from.")
    source: str = Field(..., description="Original source filename.")
    section: Optional[str] = Field(default=None, description="Source section title.")

    rule_type: RuleType = Field(default=RuleType.GENERAL_POLICY)
    rule_text: str = Field(..., description="Original or lightly cleaned rule sentence.")
    condition: Optional[str] = Field(default=None)
    expected_action: Optional[str] = Field(default=None)

    required_evidence: List[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("rule_id", "source_chunk_id", "source", "rule_text")
    @classmethod
    def rule_required_fields(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Rule fields cannot be empty.")
        return value.strip()


class Citation(BaseModel):
    """
    A required source citation for an eval case.
    """

    chunk_id: str = Field(..., description="Chunk that supports the expected answer.")
    source: str = Field(..., description="Original source filename.")
    required_evidence: str = Field(
        ...,
        description="Short explanation of what evidence this chunk provides.",
    )

    @field_validator("chunk_id", "source", "required_evidence")
    @classmethod
    def citation_fields_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Citation fields cannot be empty.")
        return value.strip()


class Rubric(BaseModel):
    """
    Weighted scoring rubric for a test case.

    We keep weights explicit so a reviewer can see how EvalForge grades answers.
    """

    faithfulness: float = Field(default=0.35, ge=0.0, le=1.0)
    policy_correctness: float = Field(default=0.30, ge=0.0, le=1.0)
    citation_accuracy: float = Field(default=0.20, ge=0.0, le=1.0)
    clarity: float = Field(default=0.10, ge=0.0, le=1.0)
    safety: float = Field(default=0.05, ge=0.0, le=1.0)
    notes: Optional[str] = None

    @model_validator(mode="after")
    def weights_should_sum_to_one(self) -> "Rubric":
        total = (
            self.faithfulness
            + self.policy_correctness
            + self.citation_accuracy
            + self.clarity
            + self.safety
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Rubric weights must sum to 1.0. Current total: {total}")
        return self


class ToolExpectation(BaseModel):
    """
    Expected behavior for tool-use tests.
    """

    expected_tool: Optional[str] = None
    expected_tool_arguments: Dict[str, Any] = Field(default_factory=dict)
    should_call_tool: bool = False
    should_ask_clarification_if_missing: List[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    """
    Main benchmark test case schema.

    This is the core output of EvalForge.
    """

    test_id: str
    project_id: str = "default_project"
    dataset_version: str = "v0.1.0"
    test_type: TestType
    risk_level: RiskLevel = RiskLevel.LOW

    user_query: str
    expected_behavior: str
    expected_answer_outline: List[str] = Field(default_factory=list)

    required_citations: List[Citation] = Field(default_factory=list)
    rubric: Rubric = Field(default_factory=Rubric)

    disallowed_behaviors: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    review_status: ReviewStatus = ReviewStatus.PENDING
    validation_errors: List[str] = Field(default_factory=list)

    tool_expectation: Optional[ToolExpectation] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("test_id", "project_id", "dataset_version")
    @classmethod
    def id_fields_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("ID fields cannot be empty.")
        return value.strip()

    @field_validator("user_query", "expected_behavior")
    @classmethod
    def text_fields_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Text fields cannot be empty.")
        return value.strip()

    @model_validator(mode="after")
    def validate_case_requirements(self) -> "EvalCase":
        citation_required_types = {
            TestType.GROUNDED_QA,
            TestType.CITATION_ALIGNMENT,
            TestType.BOUNDARY_CONDITION,
        }

        if self.test_type in citation_required_types and not self.required_citations:
            raise ValueError(
                f"{self.test_type.value} cases must include at least one required citation."
            )

        if self.test_type == TestType.TOOL_USE:
            if self.tool_expectation is None:
                raise ValueError("Tool-use test cases must include tool_expectation.")

        if self.test_type != TestType.TOOL_USE and self.tool_expectation is not None:
            raise ValueError("Only tool-use cases should include tool_expectation.")

        return self


class GenerationConfig(BaseModel):
    """
    User-selected settings for benchmark generation.
    """

    project_id: str = "support_demo"
    domain: str = "support_policy"
    dataset_version: str = "v0.1.0"

    test_types: List[TestType] = Field(
        default_factory=lambda: [
            TestType.GROUNDED_QA,
            TestType.CITATION_ALIGNMENT,
            TestType.AMBIGUITY,
            TestType.REFUSAL,
            TestType.ADVERSARIAL,
            TestType.TOOL_USE,
        ]
    )

    cases_per_chunk: int = Field(default=2, ge=1, le=10)
    require_citations: bool = True
    human_review_required: bool = True

    risk_distribution: Dict[RiskLevel, float] = Field(
        default_factory=lambda: {
            RiskLevel.LOW: 0.40,
            RiskLevel.MEDIUM: 0.40,
            RiskLevel.HIGH: 0.20,
        }
    )

    @model_validator(mode="after")
    def validate_risk_distribution(self) -> "GenerationConfig":
        total = sum(self.risk_distribution.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Risk distribution must sum to 1.0. Current total: {total}"
            )
        return self


class ValidationResult(BaseModel):
    """
    Result returned by EvalForge validators.
    """

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ExportFormat(str, Enum):
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"