from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceSpec:
    name: str
    rel_path: str
    options: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--profile",
        default="sms",
        choices=["sms", "fraud_structured", "all"],
        help="Which set of datasets to merge into one unified dataset.",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--holdout-ratio", type=float, default=0.1)
    p.add_argument("--max-source-mb", type=int, default=50)
    p.add_argument("--force-sqlite", action="store_true")
    p.add_argument(
        "--objective",
        default="f1",
        choices=["f1", "accuracy"],
        help="Metric used to pick the best model on holdout.",
    )
    p.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="Search threshold on eval split to maximize F1, then report holdout at that threshold.",
    )
    return p.parse_args()


def _default_sources(args: argparse.Namespace) -> list[SourceSpec]:
    sms_sources: list[SourceSpec] = [
        SourceSpec(
            name="smsspamcollection",
            rel_path="datasets/SMSSpamCollection",
            options={
                "dataset_name": "SMSSpamCollection",
                "train_ratio": 1.0,
                "eval_ratio": 0.0,
                "holdout_ratio": 0.0,
                "seed": args.seed,
                "channel": "sms",
            },
        ),
        SourceSpec(
            name="dataset_5971",
            rel_path="datasets/Dataset_5971.csv",
            options={
                "dataset_name": "Dataset_5971",
                "text_column": "TEXT",
                "label_column": "LABEL",
                "train_ratio": 1.0,
                "eval_ratio": 0.0,
                "holdout_ratio": 0.0,
                "seed": args.seed,
                "channel": "sms",
            },
        ),
    ]

    fraud_sources: list[SourceSpec] = [
        SourceSpec(
            name="indian_online_scam_csv",
            rel_path="datasets/Updated_Inclusive_Indian_Online_Scam_Dataset (1).csv",
            options={
                "dataset_name": "Indian Online Scam (structured)",
                "label_column": "is_fraudulent",
                "train_ratio": 1.0,
                "eval_ratio": 0.0,
                "holdout_ratio": 0.0,
                "seed": args.seed,
                "channel": "transaction",
            },
        ),
        SourceSpec(
            name="upi_transactions",
            rel_path="datasets/upi_transactions_2024.csv",
            options={
                "dataset_name": "UPI Transactions 2024",
                "label_column": "fraud_flag",
                "train_ratio": 1.0,
                "eval_ratio": 0.0,
                "holdout_ratio": 0.0,
                "seed": args.seed,
                "channel": "transaction",
            },
        ),
    ]

    if args.profile == "sms":
        return sms_sources
    if args.profile == "fraud_structured":
        return fraud_sources
    return sms_sources + fraud_sources


def _safe_size_mb(path: Path) -> float:
    return float(path.stat().st_size) / (1024.0 * 1024.0)


def _build_unified_dataset(*, repo_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    from app.schemas.dataset import DatasetUploadOptions
    from app.services.datasets import dataset_paths, ingest_dataset
    from app.services.jsonl import read_jsonl

    sources = _default_sources(args)
    combined: dict[str, dict[str, Any]] = {}
    included: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for spec in sources:
        p = (repo_root / spec.rel_path).resolve()
        if not p.exists():
            skipped.append({"source": spec.name, "path": spec.rel_path, "reason": "missing"})
            continue

        size_mb = _safe_size_mb(p)
        if size_mb > float(args.max_source_mb):
            skipped.append({"source": spec.name, "path": spec.rel_path, "reason": f"too_large_mb={size_mb:.1f}"})
            continue

        opts = DatasetUploadOptions(**spec.options)
        meta = ingest_dataset(settings=args._settings, original_filename=p.name, file_bytes=p.read_bytes(), opts=opts)
        ds_id = meta["dataset_id"]

        ds_paths = dataset_paths(args._settings, ds_id)
        rows = list(read_jsonl(ds_paths.train_path))

        for r in rows:
            text = str(r.get("text") or "").strip()
            if not text:
                continue
            label = int(r.get("label", 0))
            key = text
            prev = combined.get(key)
            if prev is None:
                combined[key] = {
                    "text": text,
                    "label": label,
                    "metadata": {
                        "source_dataset": spec.name,
                        "source_dataset_id": ds_id,
                        "source_file": p.name,
                        "channel": r.get("channel"),
                    },
                }
            else:
                prev["label"] = int(max(int(prev.get("label", 0)), label))

        included.append({"source": spec.name, "path": spec.rel_path, "dataset_id": ds_id, "num_samples": meta.get("num_samples")})

    unified_rows = list(combined.values())
    if not unified_rows:
        raise SystemExit(f"No samples produced. included={included} skipped={skipped}")

    unified_jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in unified_rows) + "\n"

    unified_opts = DatasetUploadOptions(
        dataset_name=f"unified_{args.profile}",
        train_ratio=float(args.train_ratio),
        eval_ratio=float(args.eval_ratio),
        holdout_ratio=float(args.holdout_ratio),
        seed=int(args.seed),
        channel="mixed",
    )

    unified_meta = ingest_dataset(
        settings=args._settings,
        original_filename=f"unified_{args.profile}.jsonl",
        file_bytes=unified_jsonl.encode("utf-8"),
        opts=unified_opts,
    )

    return {"unified_meta": unified_meta, "sources_included": included, "sources_skipped": skipped}


def _pick_best(records: list[dict[str, Any]], objective: str) -> dict[str, Any]:
    if not records:
        raise ValueError("No model records")

    key = "f1" if objective == "f1" else "accuracy"

    def score(r: dict[str, Any]) -> float:
        m = r.get("holdout_metrics") or {}
        return float(m.get(key, 0.0))

    return max(records, key=score)


def _threshold_search(*, probs: list[float], y_true: list[int]) -> tuple[float, dict[str, float]]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    best_thr = 0.5
    best_f1 = -1.0
    best_metrics: dict[str, float] = {}

    for i in range(5, 96):
        thr = i / 100.0
        y_pred = [1 if p >= thr else 0 for p in probs]
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1),
                "threshold": float(thr),
            }

    return best_thr, best_metrics


def _metrics_at_threshold(*, probs: list[float], y_true: list[int], thr: float) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_pred = [1 if p >= thr else 0 for p in probs]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(thr),
    }


def main() -> int:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    os.environ.setdefault("SCAMEVO_RESEARCH_MODE", "1")
    os.environ.setdefault("SCAMEVO_DO_NOT_DEPLOY", "1")

    if args.force_sqlite:
        os.environ.pop("SCAMEVO_DATABASE_URL", None)

    from app.core.config import get_settings

    get_settings.cache_clear()

    from app.core.paths import ensure_storage_layout
    from app.db.metadata import init_db

    settings = get_settings()
    setattr(args, "_settings", settings)
    ensure_storage_layout(settings)
    init_db(settings)

    unified = _build_unified_dataset(repo_root=repo_root, args=args)
    unified_meta = unified["unified_meta"]
    unified_dataset_id = unified_meta["dataset_id"]

    from app.schemas.detector import DetectorTrainRequest
    from app.services.datasets import dataset_paths
    from app.services.detector import infer_detector, train_detector
    from app.services.jsonl import read_jsonl

    ds = dataset_paths(settings, unified_dataset_id)
    eval_rows = list(read_jsonl(ds.eval_path))
    holdout_rows = list(read_jsonl(ds.holdout_path))

    grid: list[dict[str, Any]] = [
        {
            "name": "word_w12_50k_c1",
            "tfidf_analyzer": "word",
            "tfidf_ngram_min": 1,
            "tfidf_ngram_max": 2,
            "tfidf_max_features": 50000,
            "logreg_c": 1.0,
            "logreg_class_weight": None,
        },
        {
            "name": "word_w13_100k_c2_bal",
            "tfidf_analyzer": "word",
            "tfidf_ngram_min": 1,
            "tfidf_ngram_max": 3,
            "tfidf_max_features": 100000,
            "logreg_c": 2.0,
            "logreg_class_weight": "balanced",
        },
        {
            "name": "char_wb_35_200k_c2_bal",
            "tfidf_analyzer": "char_wb",
            "tfidf_ngram_min": 3,
            "tfidf_ngram_max": 5,
            "tfidf_max_features": 200000,
            "logreg_c": 2.0,
            "logreg_class_weight": "balanced",
        },
        {
            "name": "char_wb_46_200k_c4_bal",
            "tfidf_analyzer": "char_wb",
            "tfidf_ngram_min": 4,
            "tfidf_ngram_max": 6,
            "tfidf_max_features": 200000,
            "logreg_c": 4.0,
            "logreg_class_weight": "balanced",
        },
    ]

    records: list[dict[str, Any]] = []

    for cfg in grid:
        req = DetectorTrainRequest(
            dataset_id=unified_dataset_id,
            backend="tfidf_logreg",
            seed=int(args.seed),
            detection_threshold=0.5,
            tfidf_analyzer=str(cfg.get("tfidf_analyzer")),
            tfidf_ngram_min=int(cfg["tfidf_ngram_min"]),
            tfidf_ngram_max=int(cfg["tfidf_ngram_max"]),
            tfidf_max_features=int(cfg["tfidf_max_features"]),
            logreg_c=float(cfg["logreg_c"]),
            logreg_class_weight=cfg["logreg_class_weight"],
        )

        train_out = train_detector(settings=settings, req=req)
        model_id = train_out["model_id"]

        threshold = 0.5
        eval_threshold_metrics = None
        if args.optimize_threshold and eval_rows:
            eval_texts = [str(r.get("text") or "") for r in eval_rows]
            eval_y = [int(r.get("label", 0)) for r in eval_rows]
            eval_preds = infer_detector(settings=settings, model_id=model_id, texts=eval_texts, explain=False)
            eval_probs = [float(p["scam_probability"]) for p in eval_preds]
            threshold, eval_threshold_metrics = _threshold_search(probs=eval_probs, y_true=eval_y)

        holdout_metrics = {}
        if holdout_rows:
            hold_texts = [str(r.get("text") or "") for r in holdout_rows]
            hold_y = [int(r.get("label", 0)) for r in holdout_rows]
            hold_preds = infer_detector(settings=settings, model_id=model_id, texts=hold_texts, explain=False)
            hold_probs = [float(p["scam_probability"]) for p in hold_preds]
            holdout_metrics = _metrics_at_threshold(probs=hold_probs, y_true=hold_y, thr=float(threshold))

        records.append(
            {
                "name": str(cfg["name"]),
                "train": train_out,
                "eval_metrics": train_out.get("metrics", {}).get("eval"),
                "eval_threshold_metrics": eval_threshold_metrics,
                "holdout_metrics": holdout_metrics,
                "config": cfg,
            }
        )

    best = _pick_best(records, args.objective)

    out_dir = settings.storage_dir / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"best_{args.profile}_manifest.json"

    manifest = {
        "profile": args.profile,
        "objective": args.objective,
        "seed": int(args.seed),
        "unified": unified,
        "candidates": records,
        "best": best,
    }
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("BEST MODEL TRAIN COMPLETE")
    print(f"- profile:   {args.profile}")
    print(f"- dataset:   {unified_dataset_id}")
    print(f"- best:      {best.get('train', {}).get('model_id')}")
    print(f"- manifest:  {out_path}")

    return 0


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    raise SystemExit(main())
