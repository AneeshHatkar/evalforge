from pathlib import Path
from pypdf import PdfWriter

import pytest

from src.document_loader import (
    detect_source_type,
    load_document,
    load_documents_from_directory,
    make_source_id,
)
from src.schemas import SourceType


def test_detect_source_type():
    assert detect_source_type("policy.md") == SourceType.MARKDOWN
    assert detect_source_type("notes.txt") == SourceType.TEXT
    assert detect_source_type("schema.json") == SourceType.JSON
    assert detect_source_type("data.csv") == SourceType.CSV
    assert detect_source_type("unknown.docx") == SourceType.UNKNOWN

def test_detect_source_type_pdf():
    assert detect_source_type("policy.pdf") == SourceType.PDF


def test_make_source_id():
    assert make_source_id("refund_policy.md") == "refund_policy"
    assert make_source_id("Support Policy v1.txt") == "support_policy_v1"
    assert make_source_id("  Weird---File Name!!.md") == "weird_file_name"


def test_load_markdown_document(tmp_path: Path):
    sample_file = tmp_path / "refund_policy.md"
    sample_file.write_text("# Refund Policy\n\nCustomers may request refunds.", encoding="utf-8")

    document = load_document(sample_file)

    assert document.source_id == "refund_policy"
    assert document.filename == "refund_policy.md"
    assert document.source_type == SourceType.MARKDOWN
    assert "Customers may request refunds." in document.text
    assert "checksum_sha256" in document.metadata


def test_load_json_document(tmp_path: Path):
    sample_file = tmp_path / "tool_schema.json"
    sample_file.write_text('{"tool": "issue_refund"}', encoding="utf-8")

    document = load_document(sample_file)

    assert document.source_id == "tool_schema"
    assert document.source_type == SourceType.JSON
    assert '"tool": "issue_refund"' in document.text

def test_load_csv_document(tmp_path: Path):
    sample_file = tmp_path / "support_rules.csv"
    sample_file.write_text(
        "policy_area,condition,allowed_action\n"
        "refund,delivery delayed more than 7 days,issue shipping-fee refund\n",
        encoding="utf-8",
    )

    document = load_document(sample_file)

    assert document.source_id == "support_rules"
    assert document.source_type == SourceType.CSV
    assert "CSV Source" in document.text
    assert "delivery delayed more than 7 days" in document.text
    assert "issue shipping-fee refund" in document.text


def test_unsupported_file_type_raises_error(tmp_path: Path):
    sample_file = tmp_path / "policy.docx"
    sample_file.write_text("Unsupported content", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(sample_file)


def test_load_documents_from_directory(tmp_path: Path):
    (tmp_path / "a.md").write_text("# A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B text", encoding="utf-8")
    (tmp_path / "ignore.docx").write_text("Ignore me", encoding="utf-8")

    documents = load_documents_from_directory(tmp_path)

    assert len(documents) == 2
    assert {doc.filename for doc in documents} == {"a.md", "b.txt"}