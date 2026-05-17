from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Fake RAG Target")


class RAGRequest(BaseModel):
    question: str
    test_id: str | None = None
    test_type: str | None = None
    expected_citations: list[str] = []


@app.post("/answer")
def answer(request: RAGRequest) -> dict:
    question = request.question.lower()

    citations = request.expected_citations or []

    if "ignore" in question or "bypass" in question:
        answer_text = (
            "I cannot bypass the policy. I must follow the documented policy and cite the relevant source."
        )
    elif "refund" in question:
        answer_text = (
            "The customer may be eligible depending on the policy conditions, such as delay duration, compensation status, and required evidence."
        )
    elif "warranty" in question:
        answer_text = (
            "A warranty claim requires required information such as order ID, purchase date, serial number, and issue description."
        )
    elif "replacement" in question or "damaged" in question:
        answer_text = (
            "For damaged products, the customer should provide required evidence before a refund or replacement is approved."
        )
    else:
        answer_text = (
            "I need more information before making a final decision. Please provide the missing details."
        )

    return {
        "answer": answer_text,
        "citations": citations,
        "tool_calls": [],
    }