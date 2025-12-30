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
    p.add_argument(
        "--max-source-mb",
        type=int,
        default=50,
        help="Skip sources larger than this size (MB). Prevents loading huge files into memory.",
    )
    p.add_argument(
        "--include-unlabeled-as-legit",
        action="store_true",
        help="If enabled, include unlabeled datasets by setting default_label=0 (can add noise).",
    )
    p.add_argument(
        "--force-sqlite",
        action="store_true",
        help="Unset SCAMEVO_DATABASE_URL if set (use SQLite metadata).",
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

    if args.include_unlabeled_as_legit:
        sms_sources.append(
            SourceSpec(
                name="sms_data_unlabeled_legit",
                rel_path="datasets/SMS-Data.csv",
                options={
                    "dataset_name": "SMS-Data (default legit)",
                    "text_column": "text",
                    "default_label": 0,
                    "train_ratio": 1.0,
                    "eval_ratio": 0.0,
                    "holdout_ratio": 0.0,
                    "seed": args.seed,
                    "channel": "sms",
                },
            )
        )

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
    from app.schemas.dataset import DatasetUploadOptions
    from app.schemas.detector import DetectorTrainRequest
    from app.services.datasets import ingest_dataset
    from app.services.detector import train_detector
    from app.services.jsonl import read_jsonl

    get_settings.cache_clear()
    settings = get_settings()

    ensure_storage_layout(settings)
    init_db(settings)

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

        from app.services.datasets import dataset_paths

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

    unified_dataset_id = unified_meta["dataset_id"]

    train_req = DetectorTrainRequest(
        dataset_id=unified_dataset_id,
        backend="tfidf_logreg",
        seed=int(args.seed),
        detection_threshold=0.5,
    )
    train_out = train_detector(settings=settings, req=train_req)

    out_dir = settings.storage_dir / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"unified_{args.profile}_manifest.json"

    manifest = {
        "profile": args.profile,
        "created_in_storage": str(settings.storage_dir),
        "sources_included": included,
        "sources_skipped": skipped,
        "unified_dataset": unified_meta,
        "train": train_out,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("UNIFIED TRAIN COMPLETE")
    print(f"- profile:     {args.profile}")
    print(f"- dataset_id:   {unified_dataset_id}")
    print(f"- model_id:     {train_out['model_id']}")
    print(f"- manifest:     {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
