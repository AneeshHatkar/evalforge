from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.session import Base


class ProjectRecord(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    domain: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    pipeline_runs: Mapped[list["PipelineRunRecord"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )


class PipelineRunRecord(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)

    project_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("projects.project_id"),
        index=True,
    )

    dataset_version: Mapped[str] = mapped_column(String(100), default="v0.1.0")

    document_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    rule_count: Mapped[int] = mapped_column(Integer, default=0)
    case_count: Mapped[int] = mapped_column(Integer, default=0)

    quality_summary_json: Mapped[str] = mapped_column(Text)
    eval_summary_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project: Mapped["ProjectRecord"] = relationship(back_populates="pipeline_runs")
    documents: Mapped[list["DocumentRecord"]] = relationship(
        back_populates="pipeline_run",
        cascade="all, delete-orphan",
    )
    chunks: Mapped[list["ChunkRecord"]] = relationship(
        back_populates="pipeline_run",
        cascade="all, delete-orphan",
    )
    rules: Mapped[list["RuleRecord"]] = relationship(
        back_populates="pipeline_run",
        cascade="all, delete-orphan",
    )
    cases: Mapped[list["CaseRecord"]] = relationship(
        back_populates="pipeline_run",
        cascade="all, delete-orphan",
    )


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pipeline_run_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("pipeline_runs.run_id"),
        index=True,
    )

    source_id: Mapped[str] = mapped_column(String(255), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(100))
    text: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    pipeline_run: Mapped["PipelineRunRecord"] = relationship(back_populates="documents")


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pipeline_run_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("pipeline_runs.run_id"),
        index=True,
    )

    chunk_id: Mapped[str] = mapped_column(String(255), index=True)
    document_id: Mapped[str] = mapped_column(String(255), index=True)
    source: Mapped[str] = mapped_column(String(255))
    section: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    start_char: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    end_char: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text)

    pipeline_run: Mapped["PipelineRunRecord"] = relationship(back_populates="chunks")


class RuleRecord(Base):
    __tablename__ = "rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pipeline_run_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("pipeline_runs.run_id"),
        index=True,
    )

    rule_id: Mapped[str] = mapped_column(String(255), index=True)
    source_chunk_id: Mapped[str] = mapped_column(String(255), index=True)
    source: Mapped[str] = mapped_column(String(255))
    section: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    rule_type: Mapped[str] = mapped_column(String(100))
    rule_text: Mapped[str] = mapped_column(Text)
    condition: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    expected_action: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_level: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float)
    payload_json: Mapped[str] = mapped_column(Text)

    pipeline_run: Mapped["PipelineRunRecord"] = relationship(back_populates="rules")


class CaseRecord(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pipeline_run_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("pipeline_runs.run_id"),
        index=True,
    )

    test_id: Mapped[str] = mapped_column(String(255), index=True)
    test_type: Mapped[str] = mapped_column(String(100), index=True)
    risk_level: Mapped[str] = mapped_column(String(100))
    review_status: Mapped[str] = mapped_column(String(100))
    user_query: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[str] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    pipeline_run: Mapped["PipelineRunRecord"] = relationship(back_populates="cases")