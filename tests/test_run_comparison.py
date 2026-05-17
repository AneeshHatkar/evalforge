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


def run_pipeline_with_target(target_system: str):
    files = open_sample_files()

    try:
        response = client.post(
            "/pipeline/run",
            files=files,
            data={
                "project_id": "support_demo_comparison",
                "dataset_version": "v0.1.0",
                "max_cases_per_type": "4",
                "citation_support_threshold": "0.35",
                "run_eval": "true",
                "target_system": target_system,
                "pass_threshold": "0.70",
            },
        )
    finally:
        close_sample_files(files)

    assert response.status_code == 200
    return response.json()


def test_compare_pipeline_runs_good_vs_bad_target():
    bad_run = run_pipeline_with_target("intentionally_bad_target_system")
    good_run = run_pipeline_with_target("demo_target_system")

    bad_run_id = bad_run["pipeline_run_id"]
    good_run_id = good_run["pipeline_run_id"]

    response = client.get(
        f"/pipeline/runs/{good_run_id}/compare/{bad_run_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["current_run_id"] == good_run_id
    assert data["baseline_run_id"] == bad_run_id

    assert "metric_comparison" in data
    assert "average_score" in data["metric_comparison"]
    assert "pass_rate" in data["metric_comparison"]

    assert data["metric_comparison"]["average_score"]["delta"] > 0
    assert data["metric_comparison"]["pass_rate"]["delta"] >= 0

    assert data["regression_detected"] is False


def test_compare_pipeline_runs_detects_regression():
    good_run = run_pipeline_with_target("demo_target_system")
    bad_run = run_pipeline_with_target("intentionally_bad_target_system")

    good_run_id = good_run["pipeline_run_id"]
    bad_run_id = bad_run["pipeline_run_id"]

    response = client.get(
        f"/pipeline/runs/{bad_run_id}/compare/{good_run_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["current_run_id"] == bad_run_id
    assert data["baseline_run_id"] == good_run_id

    assert data["metric_comparison"]["average_score"]["delta"] < 0
    assert data["regression_detected"] is True
    assert len(data["regression_reasons"]) > 0


def test_compare_missing_run_returns_404():
    response = client.get(
        "/pipeline/runs/missing_current/compare/missing_baseline"
    )

    assert response.status_code == 404