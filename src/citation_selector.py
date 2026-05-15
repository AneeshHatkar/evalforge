from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz

from src.schemas import Chunk, Citation, EvalCase, TestType, ValidationResult


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for simple lexical matching.

    This is not semantic embedding search yet. Stage 1 uses deterministic
    string overlap and fuzzy matching.
    """

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_content_words(text: str) -> Set[str]:
    """
    Convert text into a set of useful content words.

    Stopwords are removed so overlap focuses on meaningful evidence.
    """

    normalized = normalize_for_matching(text)

    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "can",
        "for",
        "from",
        "if",
        "in",
        "is",
        "it",
        "may",
        "more",
        "must",
        "not",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "they",
        "this",
        "to",
        "was",
        "when",
        "with",
        "within",
    }

    words = {word for word in normalized.split() if len(word) > 2 and word not in stopwords}

    return words


def build_chunk_index(chunks: List[Chunk]) -> Dict[str, Chunk]:
    """
    Build a dictionary for quick citation lookup.
    """

    return {chunk.chunk_id: chunk for chunk in chunks}


def citation_exists(citation: Citation, chunk_index: Dict[str, Chunk]) -> bool:
    """
    Check whether a citation points to an existing chunk.
    """

    return citation.chunk_id in chunk_index


def get_cited_chunks(case: EvalCase, chunks: List[Chunk]) -> List[Chunk]:
    """
    Return Chunk objects cited by an EvalCase.
    """

    chunk_index = build_chunk_index(chunks)
    cited_chunks = []

    for citation in case.required_citations:
        chunk = chunk_index.get(citation.chunk_id)
        if chunk is not None:
            cited_chunks.append(chunk)

    return cited_chunks


def citation_support_score(citation: Citation, chunk: Chunk, case: Optional[EvalCase] = None) -> float:
    """
    Estimate whether a chunk supports a citation.

    Stage 1 scoring combines:
    - fuzzy similarity between required evidence and chunk text
    - content-word overlap
    - optional overlap with expected answer outline

    This is a lightweight deterministic approximation.
    """

    citation_text = citation.required_evidence
    chunk_text = chunk.text

    citation_norm = normalize_for_matching(citation_text)
    chunk_norm = normalize_for_matching(chunk_text)

    if not citation_norm or not chunk_norm:
        return 0.0

    fuzzy_score = fuzz.partial_ratio(citation_norm, chunk_norm) / 100.0

    citation_words = tokenize_content_words(citation_text)
    chunk_words = tokenize_content_words(chunk_text)

    if citation_words:
        overlap_score = len(citation_words.intersection(chunk_words)) / len(citation_words)
    else:
        overlap_score = 0.0

    outline_score = 0.0

    if case is not None and case.expected_answer_outline:
        outline_text = " ".join(case.expected_answer_outline)
        outline_words = tokenize_content_words(outline_text)

        if outline_words:
            outline_score = len(outline_words.intersection(chunk_words)) / len(outline_words)

    # Weighted simple score. Citation evidence should matter most.
    score = (0.50 * fuzzy_score) + (0.35 * overlap_score) + (0.15 * outline_score)

    return round(min(max(score, 0.0), 1.0), 4)


def citation_is_supported(
    citation: Citation,
    chunk: Chunk,
    case: Optional[EvalCase] = None,
    threshold: float = 0.35,
) -> bool:
    """
    Decide whether a citation is sufficiently supported by a chunk.
    """

    return citation_support_score(citation, chunk, case=case) >= threshold


def validate_case_citations(
    case: EvalCase,
    chunks: List[Chunk],
    support_threshold: float = 0.35,
) -> ValidationResult:
    """
    Validate that an EvalCase's required citations point to real chunks and
    roughly match the cited evidence.
    """

    errors: List[str] = []
    warnings: List[str] = []

    chunk_index = build_chunk_index(chunks)

    citation_required_types = {
        TestType.GROUNDED_QA,
        TestType.CITATION_ALIGNMENT,
        TestType.BOUNDARY_CONDITION,
        TestType.TOOL_USE,
        TestType.AMBIGUITY,
        TestType.ADVERSARIAL,
    }

    if case.test_type in citation_required_types and not case.required_citations:
        errors.append(f"{case.test_type.value} case has no required citations.")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    for citation in case.required_citations:
        chunk = chunk_index.get(citation.chunk_id)

        if chunk is None:
            errors.append(f"Citation references missing chunk_id: {citation.chunk_id}")
            continue

        if citation.source != chunk.source:
            warnings.append(
                f"Citation source mismatch for {citation.chunk_id}: "
                f"citation source={citation.source}, chunk source={chunk.source}"
            )

        score = citation_support_score(citation, chunk, case=case)

        if score < support_threshold:
            errors.append(
                f"Citation {citation.chunk_id} appears weakly supported by chunk text. "
                f"support_score={score}"
            )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def select_best_citation_for_text(
    evidence_text: str,
    chunks: List[Chunk],
    top_k: int = 1,
) -> List[Citation]:
    """
    Select best matching source chunks for a piece of evidence text.

    This is useful when a generated case is missing citations or when we
    want to repair citations.
    """

    if not evidence_text.strip():
        return []

    scored: List[Tuple[float, Chunk]] = []

    fake_citation = Citation(
        chunk_id="temporary",
        source="temporary",
        required_evidence=evidence_text,
    )

    for chunk in chunks:
        score = citation_support_score(fake_citation, chunk)

        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)

    citations: List[Citation] = []

    for score, chunk in scored[:top_k]:
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                required_evidence=evidence_text.strip(),
            )
        )

    return citations


def ensure_case_has_citations(case: EvalCase, chunks: List[Chunk]) -> EvalCase:
    """
    If a case has no citations, try to attach the best citation using its
    expected answer outline and expected behavior.

    Returns a new EvalCase object.
    """

    if case.required_citations:
        return case

    evidence_text = " ".join(case.expected_answer_outline) or case.expected_behavior

    citations = select_best_citation_for_text(evidence_text, chunks, top_k=1)

    if not citations:
        return case

    updated_data = case.model_dump()
    updated_data["required_citations"] = [citation.model_dump() for citation in citations]

    return EvalCase(**updated_data)


def validate_cases_citations(
    cases: List[EvalCase],
    chunks: List[Chunk],
    support_threshold: float = 0.35,
) -> Dict[str, ValidationResult]:
    """
    Validate citations for multiple cases.
    """

    results: Dict[str, ValidationResult] = {}

    for case in cases:
        results[case.test_id] = validate_case_citations(
            case=case,
            chunks=chunks,
            support_threshold=support_threshold,
        )

    return results