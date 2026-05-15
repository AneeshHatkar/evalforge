from __future__ import annotations

import hashlib
from importlib.resources import path
import json
import re
import pandas as pd
from pathlib import Path
from typing import Iterable, List, Union
from pypdf import PdfReader

from src.schemas import Document, SourceType


SUPPORTED_EXTENSIONS = {".md", ".txt", ".json", ".csv", ".pdf"}


def detect_source_type(filename: str) -> SourceType:
    """
    Detect the document type from the file extension.
    """

    suffix = Path(filename).suffix.lower()

    if suffix == ".md":
        return SourceType.MARKDOWN
    if suffix == ".txt":
        return SourceType.TEXT
    if suffix == ".json":
        return SourceType.JSON
    if suffix == ".csv":
        return SourceType.CSV
    if suffix == ".pdf":
        return SourceType.PDF

    return SourceType.UNKNOWN


def make_source_id(filename: str) -> str:
    """
    Create a stable, readable source ID from a filename.

    Example:
    refund_policy.md -> refund_policy
    Support Policy v1.txt -> support_policy_v1
    """

    stem = Path(filename).stem.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", stem)
    cleaned = cleaned.strip("_")

    if not cleaned:
        cleaned = "document"

    return cleaned


def compute_checksum(text: str) -> str:
    """
    Compute a SHA-256 checksum for document text.

    This helps us detect whether a source document changed later.
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """
    Normalize document text while preserving readable paragraphs.

    We keep paragraphs and headings intact because the chunker will use them.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_text_file(path: Path) -> str:
    """
    Read Markdown or TXT files safely.
    """

    return path.read_text(encoding="utf-8")


def read_json_file(path: Path) -> str:
    """
    Read JSON files and convert them into pretty text.

    For EvalForge Stage 1, this is useful for simple tool schemas.
    """

    raw_text = path.read_text(encoding="utf-8")
    data = json.loads(raw_text)

    return json.dumps(data, indent=2, ensure_ascii=False)

def read_csv_file(path: Path) -> str:
    """
    Read CSV files and convert rows into readable text.

    This lets EvalForge test simple online datasets before we build
    a database-backed ingestion pipeline.
    """

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"CSV file is empty: {path}")

    lines = []

    lines.append(f"# CSV Source: {path.name}")
    lines.append("")
    lines.append(f"Columns: {', '.join(dataframe.columns.astype(str))}")
    lines.append("")

    for index, row in dataframe.iterrows():
        lines.append(f"## Row {index + 1}")

        for column in dataframe.columns:
            value = row[column]

            if pd.isna(value):
                continue

            lines.append(f"{column}: {value}")

        lines.append("")

    return "\n".join(lines).strip()

def read_pdf_file(path: Path) -> str:
    """
    Read a PDF file and extract text page by page.

    Stage 1 PDF support is text extraction only.
    Scanned/image-only PDFs may require OCR later.
    """

    reader = PdfReader(str(path))
    pages = []

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            pages.append(f"# Page {page_index}\n\n{page_text.strip()}")

    if not pages:
        raise ValueError(
            f"No extractable text found in PDF: {path}. "
            "This may be a scanned/image-only PDF."
        )

    return "\n\n".join(pages)


def load_document(path: Union[str, Path]) -> Document:
    """
    Load a single supported document into a Document object.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported types are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    source_type = detect_source_type(path.name)

    if source_type in {SourceType.MARKDOWN, SourceType.TEXT}:
        text = read_text_file(path)
    elif source_type == SourceType.JSON:
        text = read_json_file(path)
    elif source_type == SourceType.CSV:
        text = read_csv_file(path)
    elif source_type == SourceType.PDF:
        text = read_pdf_file(path)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    normalized_text = normalize_text(text)
    source_id = make_source_id(path.name)
    checksum = compute_checksum(normalized_text)

    return Document(
        source_id=source_id,
        filename=path.name,
        source_type=source_type,
        text=normalized_text,
        metadata={
            "path": str(path),
            "extension": suffix,
            "checksum_sha256": checksum,
            "char_count": len(normalized_text),
        },
    )


def load_documents(paths: Iterable[Union[str, Path]]) -> List[Document]:
    """
    Load multiple documents.
    """

    documents = []

    for path in paths:
        document = load_document(path)
        documents.append(document)

    return documents


def load_documents_from_directory(directory: Union[str, Path]) -> List[Document]:
    """
    Load all supported documents from a directory.
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    paths = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return load_documents(paths)