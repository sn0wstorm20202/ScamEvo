from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


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
        "--cv-folds",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits for GridSearchCV.",
    )
    p.add_argument(
        "--refit",
        default="f1",
        choices=["f1", "accuracy"],
        help="Metric used to select the best params during GridSearchCV.",
    )
    p.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="Tune decision threshold on eval split for best F1, then evaluate holdout at that threshold.",
    )

    return p.parse_args()


def _safe_size_mb(path: Path) -> float:
    return float(path.stat().st_size) / (1024.0 * 1024.0)


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


def _threshold_search(*, probs: list[float], y_true: list[int]) -> tuple[float, dict[str, float]]:
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
                "f1": float(best_f1),
                "threshold": float(best_thr),
            }

    return best_thr, best_metrics


def _metrics_at_threshold(*, probs: list[float], y_true: list[int], thr: float) -> dict[str, float]:
    y_pred = [1 if p >= thr else 0 for p in probs]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(thr),
    }


def _build_unified_dataset(*, repo_root: Path, args: argparse.Namespace, settings) -> dict[str, Any]:
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
        meta = ingest_dataset(settings=settings, original_filename=p.name, file_bytes=p.read_bytes(), opts=opts)
        ds_id = meta["dataset_id"]

        ds_paths = dataset_paths(settings, ds_id)
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
        settings=settings,
        original_filename=f"unified_{args.profile}.jsonl",
        file_bytes=unified_jsonl.encode("utf-8"),
        opts=unified_opts,
    )

    return {"unified_meta": unified_meta, "sources_included": included, "sources_skipped": skipped}


def main() -> int:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    os.environ.setdefault("SCAMEVO_RESEARCH_MODE", "1")
    os.environ.setdefault("SCAMEVO_DO_NOT_DEPLOY", "1")

    if args.force_sqlite:
        os.environ.pop("SCAMEVO_DATABASE_URL", None)

    from app.core.config import get_settings
    from app.core.paths import ensure_storage_layout
    from app.db.metadata import init_db
    from app.services.datasets import dataset_paths
    from app.services.jsonl import read_jsonl

    get_settings.cache_clear()
    settings = get_settings()

    ensure_storage_layout(settings)
    init_db(settings)

    unified = _build_unified_dataset(repo_root=repo_root, args=args, settings=settings)
    unified_meta = unified["unified_meta"]
    unified_dataset_id = unified_meta["dataset_id"]

    ds = dataset_paths(settings, unified_dataset_id)
    train_rows = list(read_jsonl(ds.train_path))
    eval_rows = list(read_jsonl(ds.eval_path))
    holdout_rows = list(read_jsonl(ds.holdout_path))

    x_train = [str(r.get("text") or "") for r in train_rows]
    y_train = np.array([int(r.get("label", 0)) for r in train_rows], dtype=np.int64)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=500, solver="liblinear", random_state=int(args.seed))),
        ]
    )

    param_grid: dict[str, Any] = {
        "tfidf__analyzer": ["word", "char_wb"],
        "tfidf__ngram_range": [(1, 2), (1, 3), (3, 5), (4, 6)],
        "tfidf__max_features": [50000, 100000, 200000],
        "clf__C": [0.5, 1.0, 2.0, 4.0],
        "clf__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit=str(args.refit),
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )

    grid.fit(x_train, y_train)

    best_params = grid.best_params_
    best_cv_score = float(grid.best_score_)

    best_estimator = grid.best_estimator_

    threshold = 0.5
    eval_threshold_metrics = None

    if args.optimize_threshold and eval_rows:
        x_eval = [str(r.get("text") or "") for r in eval_rows]
        y_eval = [int(r.get("label", 0)) for r in eval_rows]
        eval_probs = best_estimator.predict_proba(x_eval)[:, 1].tolist()
        threshold, eval_threshold_metrics = _threshold_search(probs=eval_probs, y_true=y_eval)

    holdout_metrics = {}
    if holdout_rows:
        x_holdout = [str(r.get("text") or "") for r in holdout_rows]
        y_holdout = [int(r.get("label", 0)) for r in holdout_rows]
        hold_probs = best_estimator.predict_proba(x_holdout)[:, 1].tolist()
        holdout_metrics = _metrics_at_threshold(probs=hold_probs, y_true=y_holdout, thr=float(threshold))

    from app.schemas.detector import DetectorTrainRequest
    from app.services.detector import train_detector

    analyzer = str(best_params.get("tfidf__analyzer") or "word")
    ngram_range = best_params.get("tfidf__ngram_range") or (1, 2)
    max_features = int(best_params.get("tfidf__max_features") or 50000)

    req = DetectorTrainRequest(
        dataset_id=unified_dataset_id,
        backend="tfidf_logreg",
        seed=int(args.seed),
        detection_threshold=float(threshold),
        tfidf_analyzer=analyzer,
        tfidf_ngram_min=int(ngram_range[0]),
        tfidf_ngram_max=int(ngram_range[1]),
        tfidf_max_features=max_features,
        logreg_c=float(best_params.get("clf__C") or 1.0),
        logreg_class_weight=best_params.get("clf__class_weight"),
    )

    train_out = train_detector(settings=settings, req=req)

    out_dir = settings.storage_dir / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"best_cv_{args.profile}_manifest.json"

    manifest = {
        "profile": args.profile,
        "seed": int(args.seed),
        "cv_folds": int(args.cv_folds),
        "refit": str(args.refit),
        "gridsearch": {
            "best_params": best_params,
            "best_cv_score": best_cv_score,
            "best_cv_metric": str(args.refit),
        },
        "unified": unified,
        "threshold": float(threshold),
        "eval_threshold_metrics": eval_threshold_metrics,
        "holdout_metrics": holdout_metrics,
        "final_train": train_out,
    }

    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("GRIDSEARCHCV COMPLETE")
    print(f"- profile:     {args.profile}")
    print(f"- dataset_id:   {unified_dataset_id}")
    print(f"- best_model:   {train_out['model_id']}")
    print(f"- best_cv_{args.refit}: {best_cv_score}")
    print(f"- holdout_f1:   {holdout_metrics.get('f1')}")
    print(f"- holdout_acc:  {holdout_metrics.get('accuracy')}")
    print(f"- manifest:     {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
