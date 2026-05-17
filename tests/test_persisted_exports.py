from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.db.session import init_db
from backend.app.main import app
from backend.app.state import store


client = TestClient(app)


def setup_function():
    init_db()
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


def create_persisted_run() -> dict:
    files = open_sample_files()

    try:
        response = client.post(
            "/pipeline/run",
            files=files,
            data={
                "project_id": "support_demo_exports",
                "dataset_version": "v0.1.0",
                "max_cases_per_type": "3",
                "citation_support_threshold": "0.35",
                "run_eval": "true",
                "target_system": "demo_target_system",
                "pass_threshold": "0.70",
            },
        )
    finally:
        close_sample_files(files)

    assert response.status_code == 200

    return response.json()


def test_export_persisted_run_json():
    run_data = create_persisted_run()
    run_id = run_data["pipeline_run_id"]

    response = client.get(f"/pipeline/runs/{run_id}/export/json")

    assert response.status_code == 200

    data = response.json()

    assert "metadata" in data
    assert "cases" in data
    assert data["metadata"]["run_id"] == run_id
    assert len(data["cases"]) == run_data["case_count"]


def test_export_persisted_run_jsonl():
    run_data = create_persisted_run()
    run_id = run_data["pipeline_run_id"]

    response = client.get(f"/pipeline/runs/{run_id}/export/jsonl")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")

    lines = response.text.strip().splitlines()

    assert len(lines) == run_data["case_count"]
    assert "test_id" in lines[0]


def test_export_persisted_run_csv():
    run_data = create_persisted_run()
    run_id = run_data["pipeline_run_id"]

    response = client.get(f"/pipeline/runs/{run_id}/export/csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")

    content = response.text

    assert "test_id" in content
    assert "user_query" in content
    assert "expected_behavior" in content


def test_export_persisted_quality_report():
    run_data = create_persisted_run()
    run_id = run_data["pipeline_run_id"]

    response = client.get(f"/pipeline/runs/{run_id}/quality-report")

    assert response.status_code == 200

    data = response.json()

    assert data["run"]["run_id"] == run_id
    assert data["quality_summary"]["total_cases"] == run_data["case_count"]
    assert data["eval_summary"]["target_system"] == "demo_target_system"


def test_export_missing_run_returns_404():
    response = client.get("/pipeline/runs/missing_run/export/json")

    assert response.status_code == 404