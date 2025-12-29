from fastapi.testclient import TestClient

from app.main import create_app


def test_health_ok():
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "app" in body
        assert "research_mode" in body
        assert "do_not_deploy" in body
