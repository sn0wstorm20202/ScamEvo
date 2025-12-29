from fastapi.testclient import TestClient

from app.core.config import get_settings

from app.main import create_app


def test_generator_mutate_ok(monkeypatch):
    monkeypatch.setenv("SCAMEVO_RESEARCH_MODE", "1")
    monkeypatch.setenv("SCAMEVO_DO_NOT_DEPLOY", "1")
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as client:
        payload = {
            "text": "Urgent: Verify your bank account now. Click http://example.com to confirm.",
            "num_candidates": 5,
            "seed": 7,
            "similarity_threshold": 0.2,
            "require_anchors": True,
        }
        resp = client.post("/generator/mutate", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["base_text"]
        assert isinstance(data["candidates"], list)
        assert len(data["candidates"]) >= 1
        for c in data["candidates"]:
            assert isinstance(c.get("text"), str) and len(c.get("text")) > 0
            assert float(c.get("similarity")) >= 0.0
            assert "watermark" in (c.get("metadata") or {})


def test_generator_disabled_when_not_research_mode(monkeypatch):
    monkeypatch.setenv("SCAMEVO_RESEARCH_MODE", "0")
    monkeypatch.setenv("SCAMEVO_DO_NOT_DEPLOY", "1")
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as client:
        payload = {
            "text": "Urgent: Verify your bank account now. Click http://example.com to confirm.",
            "num_candidates": 3,
        }
        resp = client.post("/generator/mutate", json=payload)
        assert resp.status_code == 403
