from __future__ import annotations

from fastapi import APIRouter

from backend.app.api_schemas import LoadSampleCorpusResponse
from backend.app.state import store
from src.chunker import chunk_documents
from src.document_loader import load_documents_from_directory
from src.rule_extractor import extract_rules_from_chunks

router = APIRouter(prefix="/corpora", tags=["corpora"])


@router.post("/load-sample", response_model=LoadSampleCorpusResponse)
def load_sample_corpus() -> LoadSampleCorpusResponse:
    """
    Load sample_docs into backend memory.

    Stage 2 keeps this simple. Later we add real file upload.
    """

    store.reset_pipeline()

    documents = load_documents_from_directory("sample_docs")
    chunks = chunk_documents(documents, max_chars=700, overlap_chars=100)
    rules = extract_rules_from_chunks(chunks)

    store.documents = documents
    store.chunks = chunks
    store.rules = rules

    return LoadSampleCorpusResponse(
        document_count=len(documents),
        chunk_count=len(chunks),
        rule_count=len(rules),
    )


@router.get("/documents")
def list_documents() -> dict:
    return {
        "documents": [document.model_dump(mode="json") for document in store.documents]
    }


@router.get("/chunks")
def list_chunks() -> dict:
    return {
        "chunks": [chunk.model_dump(mode="json") for chunk in store.chunks]
    }


@router.get("/rules")
def list_rules() -> dict:
    return {
        "rules": [rule.model_dump(mode="json") for rule in store.rules]
    }