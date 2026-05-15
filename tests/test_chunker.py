from src.chunker import chunk_document, chunk_documents, split_text_by_markdown_headings
from src.schemas import Document, SourceType


def test_split_text_by_markdown_headings():
    text = """# Refund Policy

## Late Delivery Refunds

Customers can receive a shipping refund if delivery is delayed.

## Damaged Product Refunds

Customers must provide photo evidence.
"""

    sections = split_text_by_markdown_headings(text)

    assert len(sections) == 2
    assert sections[0].section_title == "Late Delivery Refunds"
    assert "shipping refund" in sections[0].text
    assert sections[1].section_title == "Damaged Product Refunds"
    assert "photo evidence" in sections[1].text


def test_chunk_document_creates_stable_chunk_ids():
    document = Document(
        source_id="refund_policy",
        filename="refund_policy.md",
        source_type=SourceType.MARKDOWN,
        text="""# Refund Policy

## Late Delivery Refunds

Customers are eligible for a shipping-fee refund if delivery is delayed by more than 7 days.

## Duplicate Refund Requests

Customers who already received compensation are not eligible for another refund.
""",
    )

    chunks = chunk_document(document, max_chars=500, overlap_chars=50)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "refund_policy_0001"
    assert chunks[1].chunk_id == "refund_policy_0002"
    assert chunks[0].source == "refund_policy.md"
    assert chunks[0].section == "Late Delivery Refunds"


def test_chunk_document_without_headings():
    document = Document(
        source_id="plain_policy",
        filename="plain_policy.txt",
        source_type=SourceType.TEXT,
        text="Customers must provide an order ID before a refund can be processed.",
    )

    chunks = chunk_document(document, max_chars=500, overlap_chars=50)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "plain_policy_0001"
    assert chunks[0].section is None
    assert "order ID" in chunks[0].text


def test_chunk_documents_multiple_documents():
    doc_a = Document(
        source_id="refund_policy",
        filename="refund_policy.md",
        source_type=SourceType.MARKDOWN,
        text="# Refund Policy\n\nCustomers may request refunds.",
    )

    doc_b = Document(
        source_id="shipping_policy",
        filename="shipping_policy.md",
        source_type=SourceType.MARKDOWN,
        text="# Shipping Policy\n\nShipping takes 5 to 7 business days.",
    )

    chunks = chunk_documents([doc_a, doc_b], max_chars=500, overlap_chars=50)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "refund_policy_0001"
    assert chunks[1].chunk_id == "shipping_policy_0001"