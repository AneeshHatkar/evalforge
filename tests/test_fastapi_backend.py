from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.state import store


client = TestClient(app)


def setup_function():
    """
    Reset backend in-memory store before each API test.

    This keeps tests independent from each other.
    """
    store.projects = {}
    store.reset_pipeline()


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["app"] == "EvalForge API"
    assert "version" in data


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "EvalForge API is running."
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"


def test_create_and_list_project():
    payload = {
        "project_id": "support_demo",
        "name": "Support Agent Benchmark",
        "domain": "support_policy",
        "description": "Benchmark for support-policy RAG and agent systems.",
    }

    response = client.post("/projects", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["project_id"] == "support_demo"
    assert data["name"] == "Support Agent Benchmark"
    assert data["domain"] == "support_policy"

    list_response = client.get("/projects")

    assert list_response.status_code == 200

    projects = list_response.json()["projects"]

    assert len(projects) == 1
    assert projects[0]["project_id"] == "support_demo"


def test_load_sample_corpus():
    response = client.post("/corpora/load-sample")

    assert response.status_code == 200

    data = response.json()

    assert data["document_count"] >= 4
    assert data["chunk_count"] > 0
    assert data["rule_count"] > 0

    docs_response = client.get("/corpora/documents")
    chunks_response = client.get("/corpora/chunks")
    rules_response = client.get("/corpora/rules")

    assert docs_response.status_code == 200
    assert chunks_response.status_code == 200
    assert rules_response.status_code == 200

    assert len(docs_response.json()["documents"]) >= 4
    assert len(chunks_response.json()["chunks"]) > 0
    assert len(rules_response.json()["rules"]) > 0


def test_generation_requires_corpus_first():
    response = client.post(
        "/generation/run",
        json={
            "project_id": "support_demo",
            "dataset_version": "v0.1.0",
            "max_cases_per_type": 4,
            "citation_support_threshold": 0.35,
        },
    )

    assert response.status_code == 400
    assert "No rules available" in response.json()["detail"]


def test_generation_after_loading_sample_corpus():
    load_response = client.post("/corpora/load-sample")

    assert load_response.status_code == 200

    response = client.post(
        "/generation/run",
        json={
            "project_id": "support_demo",
            "dataset_version": "v0.1.0",
            "max_cases_per_type": 4,
            "citation_support_threshold": 0.35,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["case_count"] > 0
    assert data["quality_summary"]["total_cases"] == data["case_count"]
    assert "test_type_distribution" in data["quality_summary"]

    cases_response = client.get("/cases")

    assert cases_response.status_code == 200

    cases_data = cases_response.json()

    assert cases_data["case_count"] == data["case_count"]
    assert len(cases_data["cases"]) == data["case_count"]


def test_get_single_case_after_generation():
    client.post("/corpora/load-sample")
    client.post(
        "/generation/run",
        json={
            "project_id": "support_demo",
            "dataset_version": "v0.1.0",
            "max_cases_per_type": 2,
            "citation_support_threshold": 0.35,
        },
    )

    cases_response = client.get("/cases")
    first_case = cases_response.json()["cases"][0]
    test_id = first_case["test_id"]

    response = client.get(f"/cases/{test_id}")

    assert response.status_code == 200

    data = response.json()

    assert data["test_id"] == test_id
    assert "user_query" in data
    assert "expected_behavior" in data


def test_get_missing_case_returns_404():
    response = client.get("/cases/does_not_exist")

    assert response.status_code == 404


def test_update_case_review_status():
    client.post("/corpora/load-sample")
    client.post(
        "/generation/run",
        json={
            "project_id": "support_demo",
            "dataset_version": "v0.1.0",
            "max_cases_per_type": 2,
            "citation_support_threshold": 0.35,
        },
    )

    cases_response = client.get("/cases")
    first_case = cases_response.json()["cases"][0]
    test_id = first_case["test_id"]

    response = client.patch(
        f"/cases/{test_id}/review",
        params={"review_status": "approved"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["updated"] is True
    assert data["case"]["review_status"] == "approved"


def test_export_json_requires_generation_first():
    response = client.get("/exports/json")

    assert response.status_code == 400
    assert "No cases available" in response.json()["detail"]


def test_export_json_after_generation():
    client.post("/corpora/load-sample")
    client.post(
        "/generation/run",
        json={
            "project_id": "support_demo",
            "dataset_version": "v0.1.0",
            "max_cases_per_type": 3,
            "citation_support_threshold": 0.35,
        },
    )

    response = client.get("/exports/json")

    assert response.status_code == 200

    data = response.json()

    assert "metadata" in data
    assert "cases" in data
    assert data["metadata"]["case_count"] == len(data["cases"])
    assert data["metadata"]["case_count"] > 0


def test_export_quality_report_after_generation():
    client.post("/corpora/load-sample")
    client.post(
        "/generation/run",
        json={
            "project_id": "support_demo",
            "dataset_version": "v0.1.0",
            "max_cases_per_type": 3,
            "citation_support_threshold": 0.35,
        },
    )

    response = client.get("/exports/quality-report")

    assert response.status_code == 200

    data = response.json()

    assert "summary" in data
    assert data["summary"]["total_cases"] > 0


def test_eval_run_requires_generation_first():
    response = client.post(
        "/eval-runs/demo",
        json={
            "target_system": "demo_target_system",
            "pass_threshold": 0.7,
        },
    )

    assert response.status_code == 400
    assert "No cases available" in response.json()["detail"]


def test_eval_run_demo_target_after_generation():
    client.post("/corpora/load-sample")
    client.post(
        "/generation/run",
        json={
            "project_id": "support_demo",
            "dataset_version": "v0.1.0",
            "max_cases_per_type": 3,
            "citation_support_threshold": 0.35,
        },
    )

    response = client.post(
        "/eval-runs/demo",
        json={
            "target_system": "demo_target_system",
            "pass_threshold": 0.7,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["result_count"] > 0
    assert data["summary"]["target_system"] == "demo_target_system"
    assert data["summary"]["total_cases"] == data["result_count"]
    assert 0.0 <= data["summary"]["average_score"] <= 1.0


def test_good_target_scores_higher_than_bad_target():
    client.post("/corpora/load-sample")
    client.post(
        "/generation/run",
        json={
            "project_id": "support_demo",
            "dataset_version": "v0.1.0",
            "max_cases_per_type": 3,
            "citation_support_threshold": 0.35,
        },
    )

    good_response = client.post(
        "/eval-runs/demo",
        json={
            "target_system": "demo_target_system",
            "pass_threshold": 0.7,
        },
    )

    bad_response = client.post(
        "/eval-runs/demo",
        json={
            "target_system": "intentionally_bad_target_system",
            "pass_threshold": 0.7,
        },
    )

    assert good_response.status_code == 200
    assert bad_response.status_code == 200

    good_score = good_response.json()["summary"]["average_score"]
    bad_score = bad_response.json()["summary"]["average_score"]

    assert good_score > bad_score


def test_get_latest_eval_run():
    client.post("/corpora/load-sample")
    client.post(
        "/generation/run",
        json={
            "project_id": "support_demo",
            "dataset_version": "v0.1.0",
            "max_cases_per_type": 3,
            "citation_support_threshold": 0.35,
        },
    )
    client.post(
        "/eval-runs/demo",
        json={
            "target_system": "demo_target_system",
            "pass_threshold": 0.7,
        },
    )

    response = client.get("/eval-runs/latest")

    assert response.status_code == 200

    data = response.json()

    assert "summary" in data
    assert "results" in data
    assert data["summary"]["total_cases"] == len(data["results"])