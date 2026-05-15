from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.state import store


client = TestClient(app)


def setup_function():
    store.projects = {}
    store.reset_pipeline()


def open_sample_files():
    paths = [
        Path("sample_docs/refund_policy.md"),
        Path("sample_docs/shipping_policy.md"),
        Path("sample_docs/warranty_policy.md"),
        Path("sample_docs/support_tool_schema.json"),
    ]

    opened_files = []

    for path in paths:
        opened_files.append(
            (
                "files",
                (
                    path.name,
                    path.open("rb"),
                    "application/octet-stream",
                ),
            )
        )

    return opened_files


def close_sample_files(files):
    for _, file_tuple in files:
        file_tuple[1].close()


def test_pipeline_run_upload_generate_validate_without_eval():
    files = open_sample_files()

    try:
        response = client.post(
            "/pipeline/run",
            files=files,
            data={
                "project_id": "support_demo",
                "dataset_version": "v0.1.0",
                "max_cases_per_type": "4",
                "citation_support_threshold": "0.35",
                "run_eval": "false",
                "target_system": "demo_target_system",
                "pass_threshold": "0.70",
            },
        )
    finally:
        close_sample_files(files)

    assert response.status_code == 200

    data = response.json()

    assert data["project_id"] == "support_demo"
    assert data["dataset_version"] == "v0.1.0"
    assert data["document_count"] == 4
    assert data["chunk_count"] > 0
    assert data["rule_count"] > 0
    assert data["case_count"] > 0
    assert data["quality_summary"]["total_cases"] == data["case_count"]
    assert data["quality_summary"]["citation_coverage"] == 1.0
    assert data["eval_summary"] is None

    cases_response = client.get("/cases")
    assert cases_response.status_code == 200
    assert cases_response.json()["case_count"] == data["case_count"]


def test_pipeline_run_with_demo_eval():
    files = open_sample_files()

    try:
        response = client.post(
            "/pipeline/run",
            files=files,
            data={
                "project_id": "support_demo",
                "dataset_version": "v0.1.0",
                "max_cases_per_type": "4",
                "citation_support_threshold": "0.35",
                "run_eval": "true",
                "target_system": "demo_target_system",
                "pass_threshold": "0.70",
            },
        )
    finally:
        close_sample_files(files)

    assert response.status_code == 200

    data = response.json()

    assert data["eval_summary"] is not None
    assert data["eval_summary"]["target_system"] == "demo_target_system"
    assert data["eval_summary"]["total_cases"] == data["case_count"]
    assert 0.0 <= data["eval_summary"]["average_score"] <= 1.0

    latest_eval_response = client.get("/eval-runs/latest")
    assert latest_eval_response.status_code == 200
    assert latest_eval_response.json()["summary"]["target_system"] == "demo_target_system"


def test_pipeline_rejects_unsupported_file_type(tmp_path):
    bad_file = tmp_path / "bad.docx"
    bad_file.write_text("unsupported", encoding="utf-8")

    with bad_file.open("rb") as file:
        response = client.post(
            "/pipeline/run",
            files=[
                (
                    "files",
                    (
                        "bad.docx",
                        file,
                        "application/octet-stream",
                    ),
                )
            ],
            data={
                "project_id": "support_demo",
                "dataset_version": "v0.1.0",
            },
        )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_pipeline_bad_target_scores_lower_than_demo_target():
    demo_files = open_sample_files()

    try:
        demo_response = client.post(
            "/pipeline/run",
            files=demo_files,
            data={
                "project_id": "support_demo",
                "dataset_version": "v0.1.0",
                "max_cases_per_type": "4",
                "citation_support_threshold": "0.35",
                "run_eval": "true",
                "target_system": "demo_target_system",
                "pass_threshold": "0.70",
            },
        )
    finally:
        close_sample_files(demo_files)

    bad_files = open_sample_files()

    try:
        bad_response = client.post(
            "/pipeline/run",
            files=bad_files,
            data={
                "project_id": "support_demo",
                "dataset_version": "v0.1.0",
                "max_cases_per_type": "4",
                "citation_support_threshold": "0.35",
                "run_eval": "true",
                "target_system": "intentionally_bad_target_system",
                "pass_threshold": "0.70",
            },
        )
    finally:
        close_sample_files(bad_files)

    assert demo_response.status_code == 200
    assert bad_response.status_code == 200

    demo_score = demo_response.json()["eval_summary"]["average_score"]
    bad_score = bad_response.json()["eval_summary"]["average_score"]

    assert demo_score > bad_score