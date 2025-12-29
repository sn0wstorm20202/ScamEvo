import json

from fastapi.testclient import TestClient

from app.main import create_app


def test_detector_tfidf_train_infer_evaluate():
    app = create_app()
    with TestClient(app) as client:
        payload = "".join(
            [
                "ham\tHey, are we still meeting today?\n",
                "spam\tUrgent: verify your bank account now, click http://example.com\n",
                "ham\tYour OTP is 123456\n",
                "spam\tWIN cash prize now, reply YES\n",
                "ham\tThanks, see you soon\n",
                "spam\tAccount will be blocked today. Confirm immediately\n",
            ]
        ).encode("utf-8")

        files = {"file": ("SMSSpamCollection", payload, "text/plain")}
        data = {"options": json.dumps({"train_ratio": 1.0, "eval_ratio": 0.0, "holdout_ratio": 0.0, "seed": 123})}

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
            "/detector/infer",
            json={
                "model_id": model_id,
                "texts": [
                    "Urgent: confirm your account now",
                    "Thanks for the update",
                ],
                "explain": True,
            },
        )
        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["model_id"] == model_id
        assert len(out["items"]) == 2
        assert all("scam_probability" in it for it in out["items"])
        assert all(it.get("token_importance") is not None for it in out["items"])

        resp = client.get(
            "/detector/evaluate",
            params={"model_id": model_id, "dataset_id": dataset_id, "split": "train", "detection_threshold": 0.5},
        )
        assert resp.status_code == 200, resp.text
        ev = resp.json()
        assert ev["model_id"] == model_id
        assert ev["dataset_id"] == dataset_id
        assert ev["split"] == "train"
        assert "metrics" in ev
