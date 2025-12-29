import json

from fastapi.testclient import TestClient

from app.main import create_app


def test_dataset_upload_summary_and_sample():
    app = create_app()
    with TestClient(app) as client:
        payload = "".join(
            [
                "ham\tHello there\n",
                "spam\tWIN cash now, reply YES\n",
                "ham\tYour OTP is 123456\n",
                "spam\tUrgent: account will be blocked today, click link\n",
            ]
        ).encode("utf-8")

        files = {"file": ("SMSSpamCollection", payload, "text/plain")}
        data = {"options": json.dumps({"train_ratio": 0.5, "eval_ratio": 0.25, "holdout_ratio": 0.25, "seed": 42})}

        resp = client.post("/dataset/upload", files=files, data=data)
        assert resp.status_code == 200, resp.text
        meta = resp.json()

        dataset_id = meta["dataset_id"]
        assert meta["num_samples"] == 4
        assert meta["num_scam"] == 2
        assert meta["num_legit"] == 2

        resp = client.get("/dataset/summary", params={"dataset_id": dataset_id})
        assert resp.status_code == 200
        summary = resp.json()
        assert summary["dataset_id"] == dataset_id

        resp = client.get("/dataset/sample", params={"dataset_id": dataset_id, "split": "train", "n": 2})
        assert resp.status_code == 200
        samples = resp.json()["samples"]
        assert len(samples) == 2
        assert all("text" in s and "label" in s for s in samples)


def test_dataset_smish_collection_txt():
    app = create_app()
    with TestClient(app) as client:
        payload = "".join(
            [
                "ham\tHello there\n",
                "smish\tVerify account now at http://x.y\n",
            ]
        ).encode("utf-8")

        files = {"file": ("SMSSmishCollection.txt", payload, "text/plain")}
        data = {"options": json.dumps({"train_ratio": 1.0, "eval_ratio": 0.0, "holdout_ratio": 0.0, "seed": 42})}

        resp = client.post("/dataset/upload", files=files, data=data)
        assert resp.status_code == 200, resp.text
        meta = resp.json()
        assert meta["num_samples"] == 2
        assert meta["num_scam"] == 1
        assert meta["num_legit"] == 1


def test_dataset_unlabeled_csv_with_default_label_and_multiline_text():
    app = create_app()
    with TestClient(app) as client:
        payload = "".join(
            [
                "phoneNumber,id,updateAt,senderAddress,text\n",
                "x1,uuid1,2022-01-01,SENDER,\"Line1\\nLine2\"\n",
                "x2,uuid2,2022-01-02,SENDER2,Hello\n",
            ]
        ).encode("utf-8")

        files = {"file": ("SMS-Data.csv", payload, "text/csv")}
        data = {"options": json.dumps({"default_label": 0, "train_ratio": 1.0, "eval_ratio": 0.0, "holdout_ratio": 0.0, "seed": 42})}

        resp = client.post("/dataset/upload", files=files, data=data)
        assert resp.status_code == 200, resp.text
        meta = resp.json()
        assert meta["num_samples"] == 2
        assert meta["num_scam"] == 0
        assert meta["num_legit"] == 2


def test_dataset_structured_fraud_csv_synthesized_text():
    app = create_app()
    with TestClient(app) as client:
        payload = "".join(
            [
                "transaction_id,customer_id,amount,location,is_fraudulent,fraud_type\n",
                "1,10,1000,Delhi,0,scam\n",
                "2,11,2500,Mumbai,1,phishing\n",
            ]
        ).encode("utf-8")

        files = {"file": ("fraud.csv", payload, "text/csv")}
        data = {"options": json.dumps({"train_ratio": 1.0, "eval_ratio": 0.0, "holdout_ratio": 0.0, "seed": 42})}

        resp = client.post("/dataset/upload", files=files, data=data)
        assert resp.status_code == 200, resp.text
        meta = resp.json()
        assert meta["num_samples"] == 2
        assert meta["num_scam"] == 1
        assert meta["num_legit"] == 1

        dataset_id = meta["dataset_id"]
        resp = client.get("/dataset/sample", params={"dataset_id": dataset_id, "split": "train", "n": 2})
        assert resp.status_code == 200
        samples = resp.json()["samples"]
        assert all(isinstance(s.get("text"), str) and len(s.get("text")) > 0 for s in samples)
