from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.schemas import Chunk, Document


@dataclass
class SectionBlock:
    """
    Internal helper object representing a section of a document.

    We use this before creating final EvalForge Chunk objects.
    """

    section_title: Optional[str]
    text: str
    start_char: int
    end_char: int


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def clean_section_title(title: Optional[str]) -> Optional[str]:
    """
    Normalize section titles.

    Example:
    '## Late Delivery Refunds   ' -> 'Late Delivery Refunds'
    """

    if title is None:
        return None

    title = title.strip()
    title = re.sub(r"\s+", " ", title)

    return title or None


def split_text_by_markdown_headings(text: str) -> List[SectionBlock]:
    """
    Split document text into section blocks using Markdown headings.

    If no headings are found, the whole document becomes one section.
    """

    matches = list(HEADING_PATTERN.finditer(text))

    if not matches:
        return [
            SectionBlock(
                section_title=None,
                text=text.strip(),
                start_char=0,
                end_char=len(text),
            )
        ]

    sections: List[SectionBlock] = []

    for index, match in enumerate(matches):
        heading_title = clean_section_title(match.group(2))

        section_start = match.end()
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)

        section_text = text[section_start:section_end].strip()

        # If a heading has no content under it, skip it.
        if not section_text:
            continue

        sections.append(
            SectionBlock(
                section_title=heading_title,
                text=section_text,
                start_char=section_start,
                end_char=section_end,
            )
        )

    if not sections:
        return [
            SectionBlock(
                section_title=None,
                text=text.strip(),
                start_char=0,
                end_char=len(text),
            )
        ]

    return sections


def split_long_text(
    text: str,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[Tuple[str, int, int]]:
    """
    Split long text into smaller pieces.

    We prefer paragraph boundaries. If a paragraph is too long, we fall back
    to character windows with overlap.
    """

    text = text.strip()

    if len(text) <= max_chars:
        return [(text, 0, len(text))]

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]

    chunks: List[Tuple[str, int, int]] = []
    current_parts: List[str] = []
    current_start: Optional[int] = None
    search_offset = 0

    for paragraph in paragraphs:
        paragraph_start = text.find(paragraph, search_offset)
        paragraph_end = paragraph_start + len(paragraph)
        search_offset = paragraph_end

        # If one paragraph is huge, split it directly.
        if len(paragraph) > max_chars:
            if current_parts:
                current_text = "\n\n".join(current_parts).strip()
                if current_text and current_start is not None:
                    current_end = current_start + len(current_text)
                    chunks.append((current_text, current_start, current_end))
                current_parts = []
                current_start = None

            chunks.extend(split_by_character_window(paragraph, max_chars, overlap_chars, paragraph_start))
            continue

        candidate_text = "\n\n".join(current_parts + [paragraph]).strip()

        if current_parts and len(candidate_text) > max_chars:
            current_text = "\n\n".join(current_parts).strip()
            if current_text and current_start is not None:
                current_end = current_start + len(current_text)
                chunks.append((current_text, current_start, current_end))

            current_parts = [paragraph]
            current_start = paragraph_start
        else:
            if current_start is None:
                current_start = paragraph_start
            current_parts.append(paragraph)

    if current_parts and current_start is not None:
        current_text = "\n\n".join(current_parts).strip()
        current_end = current_start + len(current_text)
        chunks.append((current_text, current_start, current_end))

    return chunks


def split_by_character_window(
    text: str,
    max_chars: int,
    overlap_chars: int,
    base_offset: int = 0,
) -> List[Tuple[str, int, int]]:
    """
    Fallback splitter for very long paragraphs.

    This keeps chunks bounded even when the source has no paragraph breaks.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0.")

    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars.")

    chunks: List[Tuple[str, int, int]] = []

    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                (
                    chunk_text,
                    base_offset + start,
                    base_offset + end,
                )
            )

        if end == len(text):
            break

        start = max(0, end - overlap_chars)

    return chunks


def make_chunk_id(document_id: str, chunk_number: int) -> str:
    """
    Create stable chunk IDs.

    Example:
    refund_policy + 1 -> refund_policy_0001
    """

    return f"{document_id}_{chunk_number:04d}"


def chunk_document(
    document: Document,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[Chunk]:
    """
    Convert a Document into citation-ready Chunks.
    """

    if max_chars <= 100:
        raise ValueError("max_chars should be greater than 100 for useful chunks.")

    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative.")

    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars.")

    section_blocks = split_text_by_markdown_headings(document.text)
    chunks: List[Chunk] = []
    chunk_counter = 1

    for section in section_blocks:
        split_parts = split_long_text(
            section.text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        for chunk_text, relative_start, relative_end in split_parts:
            absolute_start = section.start_char + relative_start
            absolute_end = section.start_char + relative_end

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.source_id, chunk_counter),
                    document_id=document.source_id,
                    source=document.filename,
                    section=section.section_title,
                    text=chunk_text,
                    start_char=absolute_start,
                    end_char=absolute_end,
                    metadata={
                        "source_type": document.source_type.value,
                        "document_checksum": document.metadata.get("checksum_sha256"),
                        "char_count": len(chunk_text),
                    },
                )
            )

            chunk_counter += 1

    return chunks


def chunk_documents(
    documents: List[Document],
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[Chunk]:
    """
    Chunk multiple documents.
    """

    all_chunks: List[Chunk] = []

    for document in documents:
        all_chunks.extend(
            chunk_document(
                document=document,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )

    return all_chunks