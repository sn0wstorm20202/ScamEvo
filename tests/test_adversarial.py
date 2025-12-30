import json

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app


def test_adversarial_run_history_and_robustness(monkeypatch):
    monkeypatch.setenv("SCAMEVO_RESEARCH_MODE", "1")
    monkeypatch.setenv("SCAMEVO_DO_NOT_DEPLOY", "1")
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as client:
        payload = "".join(
            [
                "ham\tHello there\n",
                "spam\tUrgent: verify your bank account now, click http://example.com\n",
                "ham\tYour OTP is 123456\n",
                "spam\tWIN cash prize now, reply YES\n",
            ]
        ).encode("utf-8")

        files = {"file": ("SMSSpamCollection", payload, "text/plain")}
        data = {"options": json.dumps({"train_ratio": 1.0, "eval_ratio": 0.0, "holdout_ratio": 0.0, "seed": 42})}

        resp = client.post("/dataset/upload", files=files, data=data)
        assert resp.status_code == 200, resp.text
        dataset_id = resp.json()["dataset_id"]

        req = {
            "dataset_id": dataset_id,
            "split": "train",
            "rounds": 1,
            "seeds_per_round": 2,
            "candidates_per_seed": 2,
            "similarity_threshold": 0.0,
            "require_anchors": False,
            "detection_threshold": 0.5,
            "seed": 9,
            "dry_run": True,
        }

        resp = client.post("/adversarial/run", json=req)
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["run_id"]
        assert isinstance(run_id, str) and len(run_id) > 0

        resp = client.get("/adversarial/history", params={"limit": 10})
        assert resp.status_code == 200, resp.text
        runs = resp.json()["runs"]
        assert any(r["id"] == run_id for r in runs)

        resp = client.get("/robustness/report", params={"run_id": run_id})
        assert resp.status_code == 200, resp.text
        report = resp.json()
        assert report["run_id"] == run_id
        assert "metrics" in report


def test_adversarial_retrain_dry_run(monkeypatch):
    monkeypatch.setenv("SCAMEVO_RESEARCH_MODE", "1")
    monkeypatch.setenv("SCAMEVO_DO_NOT_DEPLOY", "1")
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as client:
        payload = "".join(
            [
                "ham\tHello there\n",
                "spam\tUrgent: verify your bank account now, click http://example.com\n",
                "ham\tYour OTP is 123456\n",
                "spam\tWIN cash prize now, reply YES\n",
            ]
        ).encode("utf-8")

        files = {"file": ("SMSSpamCollection", payload, "text/plain")}
        data = {"options": json.dumps({"train_ratio": 1.0, "eval_ratio": 0.0, "holdout_ratio": 0.0, "seed": 42})}

        resp = client.post("/dataset/upload", files=files, data=data)
        assert resp.status_code == 200, resp.text
        dataset_id = resp.json()["dataset_id"]

        resp = client.post(
            "/detector/train",
            json={
                "dataset_id": dataset_id,
                "backend": "tfidf_logreg",
                "detection_threshold": 0.5,
                "seed": 123,
            },
        )
        assert resp.status_code == 200, resp.text
        model_id = resp.json()["model_id"]

        resp = client.post(
            "/adversarial/retrain",
            json={
                "dataset_id": dataset_id,
                "model_id": model_id,
                "split": "train",
                "rounds": 1,
                "seeds_per_round": 2,
                "candidates_per_seed": 2,
                "similarity_threshold": 0.0,
                "require_anchors": False,
                "hard_max_examples": 5,
                "dry_run": True,
            },
        )
        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["run_type"] == "adversarial_retrain"
        assert isinstance(out["run_id"], str) and len(out["run_id"]) > 0
        assert isinstance(out.get("artifacts_dir"), str) and len(out.get("artifacts_dir")) > 0
