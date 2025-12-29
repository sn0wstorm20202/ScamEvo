from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to dataset file relative to repo root")
    p.add_argument("--dataset-name", default="demo_dataset")
    p.add_argument("--storage-dir", default=None, help="Override SCAMEVO_STORAGE_DIR")
    p.add_argument("--db-path", default=None, help="Override SCAMEVO_DB_PATH")
    p.add_argument("--force-sqlite", action="store_true", help="Unset SCAMEVO_DATABASE_URL if set")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    dataset_path = (repo_root / args.dataset).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    os.environ["SCAMEVO_RESEARCH_MODE"] = "1"
    os.environ["SCAMEVO_DO_NOT_DEPLOY"] = "1"
    os.environ["SCAMEVO_DEMO_MODE"] = "1"

    if args.storage_dir:
        os.environ["SCAMEVO_STORAGE_DIR"] = str((repo_root / args.storage_dir).resolve())
    if args.db_path:
        os.environ["SCAMEVO_DB_PATH"] = str((repo_root / args.db_path).resolve())
    else:
        os.environ["SCAMEVO_DB_PATH"] = str((repo_root / "storage" / "demo_metadata.sqlite3").resolve())

    if args.force_sqlite and "SCAMEVO_DATABASE_URL" in os.environ:
        os.environ.pop("SCAMEVO_DATABASE_URL", None)

    from app.core.config import get_settings

    get_settings.cache_clear()

    from app.main import create_app

    app = create_app()

    manifest_dir = get_settings().storage_dir / "demo"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    with TestClient(app) as client:
        file_bytes = dataset_path.read_bytes()

        upload_resp = client.post(
            "/dataset/upload",
            files={"file": (dataset_path.name, file_bytes, "application/octet-stream")},
            data={"options": json.dumps({"dataset_name": args.dataset_name})},
        )
        if upload_resp.status_code != 200:
            raise SystemExit(f"Dataset upload failed: {upload_resp.status_code} {upload_resp.text}")
        dataset_meta = upload_resp.json()
        dataset_id = dataset_meta["dataset_id"]

        train_resp = client.post(
            "/detector/train",
            json={"dataset_id": dataset_id},
        )
        if train_resp.status_code != 200:
            raise SystemExit(f"Detector train failed: {train_resp.status_code} {train_resp.text}")
        train_out = train_resp.json()
        model_id = train_out["model_id"]

        eval_resp = client.get(
            "/detector/evaluate",
            params={"model_id": model_id, "dataset_id": dataset_id},
        )
        if eval_resp.status_code != 200:
            raise SystemExit(f"Detector evaluate failed: {eval_resp.status_code} {eval_resp.text}")
        eval_out = eval_resp.json()

        adv_resp = client.post(
            "/adversarial/run",
            json={"dataset_id": dataset_id, "model_id": model_id},
        )
        if adv_resp.status_code != 200:
            raise SystemExit(f"Adversarial run failed: {adv_resp.status_code} {adv_resp.text}")
        adv_out = adv_resp.json()
        run_id = adv_out["run_id"]

        report_resp = client.get("/robustness/report", params={"run_id": run_id})
        if report_resp.status_code != 200:
            raise SystemExit(f"Robustness report failed: {report_resp.status_code} {report_resp.text}")
        report_out = report_resp.json()

    artifacts_dir = Path(adv_out["artifacts_dir"]).resolve()
    robustness_path = artifacts_dir / "robustness_report.json"
    robustness_path.write_text(json.dumps(report_out, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "dataset_file": str(dataset_path),
        "dataset_id": dataset_id,
        "dataset_meta": dataset_meta,
        "model_id": model_id,
        "train": train_out,
        "evaluate": eval_out,
        "run_id": run_id,
        "adversarial": adv_out,
        "robustness_report_path": str(robustness_path),
        "artifacts_dir": str(artifacts_dir),
    }

    manifest_path = manifest_dir / "demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DEMO COMPLETE")
    print(f"- dataset_id: {dataset_id}")
    print(f"- model_id:   {model_id}")
    print(f"- run_id:     {run_id}")
    print(f"- manifest:   {manifest_path}")
    print(f"- run dir:    {artifacts_dir}")
    print(f"- report:     {robustness_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
