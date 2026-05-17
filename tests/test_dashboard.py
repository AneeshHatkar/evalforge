from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_dashboard_page_loads():
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "EvalForge Dashboard" in response.text
    assert "Run Pipeline" in response.text
    assert "Generated Cases Preview" in response.text
    assert "Persisted Runs" in response.text
    assert "Run Comparison" in response.text