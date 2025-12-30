from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        default="datasets/SMSSpamCollection",
        help="Path to dataset file relative to repo root",
    )
    p.add_argument("--dataset-name", default="demo_dataset")
    p.add_argument("--storage-dir", default=None, help="Override SCAMEVO_STORAGE_DIR")
    p.add_argument("--db-path", default=None, help="Override SCAMEVO_DB_PATH")
    p.add_argument("--force-sqlite", action="store_true", help="Unset SCAMEVO_DATABASE_URL if set")
    return p.parse_args()


def _safe_get(d: dict, *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "n/a"


def main() -> int:
    args = _parse_args()
    t0 = time.time()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    dataset_path = (repo_root / args.dataset).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    os.environ["SCAMEVO_RESEARCH_MODE"] = "1"
    os.environ["SCAMEVO_DO_NOT_DEPLOY"] = "1"
    os.environ["SCAMEVO_DEMO_MODE"] = "1"
    os.environ["SCAMEVO_GENERATOR_BACKEND"] = "rule"

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

    demo_root = get_settings().storage_dir / "demo"
    demo_root.mkdir(parents=True, exist_ok=True)

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
            json={"dataset_id": dataset_id, "backend": "tfidf_logreg", "detection_threshold": 0.5, "seed": 1337},
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
        attack_run_id = adv_out["run_id"]

        attack_report_resp = client.get("/robustness/report", params={"run_id": attack_run_id})
        if attack_report_resp.status_code != 200:
            raise SystemExit(f"Robustness report failed: {attack_report_resp.status_code} {attack_report_resp.text}")
        attack_report_out = attack_report_resp.json()

        retrain_resp = client.post(
            "/adversarial/retrain",
            json={"dataset_id": dataset_id, "model_id": model_id},
        )
        if retrain_resp.status_code != 200:
            raise SystemExit(f"Adversarial retrain failed: {retrain_resp.status_code} {retrain_resp.text}")
        retrain_out = retrain_resp.json()
        retrain_run_id = retrain_out["run_id"]

        retrain_report_resp = client.get("/robustness/report", params={"run_id": retrain_run_id})
        if retrain_report_resp.status_code != 200:
            raise SystemExit(f"Robustness report failed: {retrain_report_resp.status_code} {retrain_report_resp.text}")
        retrain_report_out = retrain_report_resp.json()

    attack_artifacts_dir = Path(adv_out["artifacts_dir"]).resolve()
    retrain_artifacts_dir = Path(retrain_out["artifacts_dir"]).resolve()

    demo_dir = demo_root / str(retrain_run_id)
    demo_dir.mkdir(parents=True, exist_ok=True)

    attack_report_path = demo_dir / "robustness_attack_report.json"
    attack_report_path.write_text(json.dumps(attack_report_out, ensure_ascii=False, indent=2), encoding="utf-8")

    robustness_path = demo_dir / "robustness_report.json"
    robustness_path.write_text(json.dumps(retrain_report_out, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = demo_dir / "experiment_manifest.json"
    summary_path = demo_dir / "summary.md"

    baseline_metrics = _safe_get(eval_out, "metrics", default={})
    attacked_evasion = _safe_get(attack_report_out, "metrics", "evasion_rate")
    retrain_internal_evasion = _safe_get(retrain_report_out, "metrics", "evasion_rate")
    defended_attack_evasion = _safe_get(retrain_report_out, "metrics", "defended_attack", "evasion_rate")
    demo_delta_evasion: float | None
    try:
        if attacked_evasion is None or defended_attack_evasion is None:
            demo_delta_evasion = None
        else:
            demo_delta_evasion = float(attacked_evasion) - float(defended_attack_evasion)
    except Exception:
        demo_delta_evasion = None
    defended_f1_minus_baseline_f1 = _safe_get(retrain_report_out, "metrics", "delta", "defended_f1_minus_baseline_f1")

    summary_md = "\n".join(
        [
            "# SCAM-EVO Demo Story (3 slides)",
            "",
            "## Slide 1 — Baseline detector",
            f"- Dataset: `{dataset_path.name}`",
            f"- dataset_id: `{dataset_id}`",
            f"- Model: TF-IDF + Logistic Regression (`{model_id}`)",
            f"- Holdout accuracy: `{_safe_get(baseline_metrics, 'accuracy')}`",
            f"- Holdout precision: `{_safe_get(baseline_metrics, 'precision')}`",
            f"- Holdout recall: `{_safe_get(baseline_metrics, 'recall')}`",
            f"- Holdout F1: `{_safe_get(baseline_metrics, 'f1')}`",
            "",
            "## Slide 2 — Attacked (rule-based adversary)",
            "- Generator backend: `rule`",
            f"- attack_run_id: `{attack_run_id}`",
            f"- Evasion rate: `{_fmt_pct(attacked_evasion)}`",
            "",
            "## Slide 3 — Defended (adversarial retraining)",
            f"- retrain_run_id: `{retrain_run_id}`",
            f"- Retrain internal attack evasion rate (base model): `{_fmt_pct(retrain_internal_evasion)}`",
            f"- Defended attack evasion rate: `{_fmt_pct(defended_attack_evasion)}`",
            f"- Delta (Slide2 attacked - defended_attack) evasion rate: `{_fmt_pct(demo_delta_evasion)}`",
            f"- Delta defended F1 - baseline F1: `{defended_f1_minus_baseline_f1}`",
            "",
            "## Artifacts",
            f"- experiment_manifest.json: `{manifest_path}`",
            f"- robustness_report.json: `{robustness_path}`",
            f"- summary.md: `{summary_path}`",
        ]
    )
    summary_path.write_text(summary_md, encoding="utf-8")

    manifest = {
        "generator_backend": "rule",
        "demo_dir": str(demo_dir),
        "dataset_file": str(dataset_path),
        "dataset_id": dataset_id,
        "dataset_meta": dataset_meta,
        "model_id": model_id,
        "train": train_out,
        "evaluate": eval_out,
        "attack_run_id": attack_run_id,
        "adversarial": adv_out,
        "attack_artifacts_dir": str(attack_artifacts_dir),
        "attack_robustness_report_path": str(attack_report_path),
        "retrain_run_id": retrain_run_id,
        "adversarial_retrain": retrain_out,
        "retrain_artifacts_dir": str(retrain_artifacts_dir),
        "robustness_report_path": str(robustness_path),
        "summary_md_path": str(summary_path),
        "runtime_seconds": float(time.time() - t0),
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DEMO COMPLETE")
    print(f"- dataset_id: {dataset_id}")
    print(f"- model_id:   {model_id}")
    print(f"- attack_run_id:  {attack_run_id}")
    print(f"- retrain_run_id: {retrain_run_id}")
    print(f"- demo_dir:       {demo_dir}")
    print(f"- manifest:       {manifest_path}")
    print(f"- report:         {robustness_path}")
    print(f"- summary:        {summary_path}")
    print(f"- runtime_sec:    {manifest['runtime_seconds']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
