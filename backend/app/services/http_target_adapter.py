from __future__ import annotations

import time
from typing import Any, Dict, List

import requests

from backend.app.api_schemas import HttpTargetConfig
from src.schemas import EvalCase, TargetResponse


def extract_nested_value(data: Dict[str, Any], field_path: str, default=None):
    """
    Extract a value from a response dictionary.

    Supports simple dotted paths:
    answer
    data.answer
    result.citations
    """

    current = data

    for part in field_path.split("."):
        if not isinstance(current, dict):
            return default

        if part not in current:
            return default

        current = current[part]

    return current


def normalize_citations(value: Any) -> List[str]:
    """
    Normalize target API citations into a list of strings.

    Supports:
    - ["chunk_1", "chunk_2"]
    - [{"chunk_id": "chunk_1"}, {"source_id": "chunk_2"}]
    - "chunk_1"
    """

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        citations = []

        for item in value:
            if isinstance(item, str):
                citations.append(item)
            elif isinstance(item, dict):
                if "chunk_id" in item:
                    citations.append(str(item["chunk_id"]))
                elif "source_id" in item:
                    citations.append(str(item["source_id"]))
                elif "id" in item:
                    citations.append(str(item["id"]))

        return citations

    return []


def normalize_tool_calls(value: Any) -> List[Dict[str, Any]]:
    """
    Normalize target API tool calls.

    Expected shape:
    [
      {
        "name": "issue_shipping_refund",
        "arguments": {...}
      }
    ]
    """

    if value is None:
        return []

    if not isinstance(value, list):
        return []

    normalized = []

    for item in value:
        if not isinstance(item, dict):
            continue

        name = item.get("name") or item.get("tool") or item.get("tool_name")
        arguments = item.get("arguments") or item.get("args") or {}

        if name:
            normalized.append(
                {
                    "name": name,
                    "arguments": arguments if isinstance(arguments, dict) else {},
                }
            )

    return normalized


def build_http_target_system(config: HttpTargetConfig):
    """
    Build a target-system callable compatible with EvalForge's eval runner.

    The returned function has this shape:
    target_system(user_query: str, case: EvalCase) -> TargetResponse
    """

    if config.method.upper() != "POST":
        raise ValueError("Stage 2 HTTP target adapter only supports POST.")

    def target_system(user_query: str, case: EvalCase) -> TargetResponse:
        start = time.perf_counter()

        payload = {
            config.request_field: user_query,
            "test_id": case.test_id,
            "test_type": case.test_type.value,
            "expected_citations": [
                citation.chunk_id for citation in case.required_citations
            ],
        }

        try:
            response = requests.post(
                config.target_url,
                json=payload,
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            response_data = response.json()

            answer = extract_nested_value(
                response_data,
                config.response_answer_field,
                default="",
            )

            raw_citations = (
                extract_nested_value(
                    response_data,
                    config.response_citations_field,
                    default=[],
                )
                if config.response_citations_field
                else []
            )

            raw_tool_calls = (
                extract_nested_value(
                    response_data,
                    config.response_tool_calls_field,
                    default=[],
                )
                if config.response_tool_calls_field
                else []
            )

            latency_ms = (time.perf_counter() - start) * 1000

            return TargetResponse(
                answer=str(answer or ""),
                citations=normalize_citations(raw_citations),
                tool_calls=normalize_tool_calls(raw_tool_calls),
                latency_ms=latency_ms,
                metadata={
                    "target_url": config.target_url,
                    "status_code": response.status_code,
                    "adapter": "http_target_adapter",
                },
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000

            return TargetResponse(
                answer="",
                citations=[],
                tool_calls=[],
                latency_ms=latency_ms,
                metadata={
                    "target_url": config.target_url,
                    "adapter": "http_target_adapter",
                    "error": str(exc),
                },
            )

    return target_system