from pathlib import Path

from fastapi.testclient import TestClient
from backend.app.db.session import init_db
from backend.app.main import app
from backend.app.state import store


client = TestClient(app)


def setup_function():
    """
    Reset in-memory store and ensure SQLite tables exist before each test.
    """

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


def test_pipeline_run_is_persisted_to_sqlite():
    files = open_sample_files()

    try:
        response = client.post(
            "/pipeline/run",
            files=files,
            data={
                "project_id": "support_demo_persistence",
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

    data = response.json()

    assert data["pipeline_run_id"] is not None
    assert data["case_count"] > 0

    run_id = data["pipeline_run_id"]

    list_response = client.get("/pipeline/runs")

    assert list_response.status_code == 200
    assert list_response.json()["run_count"] >= 1

    get_response = client.get(f"/pipeline/runs/{run_id}")

    assert get_response.status_code == 200

    run_data = get_response.json()

    assert run_data["run_id"] == run_id
    assert run_data["project_id"] == "support_demo_persistence"
    assert run_data["case_count"] == data["case_count"]

    cases_response = client.get(f"/pipeline/runs/{run_id}/cases")

    assert cases_response.status_code == 200

    cases_data = cases_response.json()

    assert cases_data["run_id"] == run_id
    assert cases_data["case_count"] == data["case_count"]
    assert len(cases_data["cases"]) == data["case_count"]