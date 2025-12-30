from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from app.core.config import Settings
from app.db.metadata import insert_run
from app.schemas.adversarial import AdversarialRetrainRequest, AdversarialRunRequest
from app.schemas.detector import DetectorTrainRequest
from app.services.datasets import create_augmented_dataset, dataset_paths
from app.services.detector import evaluate_model_on_dataset, infer_detector, load_model_meta, load_persisted_threshold, train_detector
from app.services.generator import mutate_text_with_backend
from app.services.jsonl import read_jsonl, write_jsonl


@dataclass(frozen=True)
class RunPaths:
    root_dir: Path
    config_path: Path
    summary_path: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_paths(settings: Settings, run_id: str) -> RunPaths:
    root = settings.runs_dir / run_id
    return RunPaths(root_dir=root, config_path=root / "run_config.json", summary_path=root / "summary.json")


def _pick_seed_rows(
    *,
    rows: list[dict[str, Any]],
    seeds_per_round: int,
    seed: int,
) -> list[dict[str, Any]]:
    scams = [r for r in rows if int(r.get("label", 0)) == 1]
    pool = scams if scams else rows
    rnd = random.Random(seed)
    if len(pool) <= seeds_per_round:
        return pool
    return rnd.sample(pool, k=seeds_per_round)


def _score_candidates(
    *,
    settings: Settings,
    model_id: str | None,
    texts: list[str],
    detection_threshold: float,
    dry_run: bool,
    seed: int,
) -> tuple[list[float], list[bool]]:
    if dry_run or not model_id:
        rnd = random.Random(seed)
        scores = [float(rnd.random()) for _ in texts]
        evasive = [s < float(detection_threshold) for s in scores]
        return scores, evasive

    preds = infer_detector(settings=settings, model_id=model_id, texts=texts, explain=False)
    scores = [float(p["scam_probability"]) for p in preds]
    evasive = [s < float(detection_threshold) for s in scores]
    return scores, evasive


def _binary_metrics(*, y_true: list[int], probs: list[float], detection_threshold: float) -> dict[str, Any]:
    if not y_true:
        return {"num_samples": 0, "threshold": float(detection_threshold)}
    yt = np.array([int(v) for v in y_true], dtype=np.int64)
    yp = (np.array([float(p) for p in probs], dtype=np.float64) >= float(detection_threshold)).astype(np.int64)
    return {
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "threshold": float(detection_threshold),
    }


def run_adversarial_retrain(*, settings: Settings, req: AdversarialRetrainRequest) -> dict[str, Any]:
    ds = dataset_paths(settings, req.dataset_id)
    path = {"train": ds.train_path, "eval": ds.eval_path, "holdout": ds.holdout_path}[req.split]
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError("Selected dataset split is empty")

    run_id = str(uuid.uuid4())
    created_at = _utc_now_iso()
    run_type = "adversarial_retrain"

    paths = run_paths(settings, run_id)
    paths.root_dir.mkdir(parents=True, exist_ok=True)

    config = req.model_dump()
    config["run_id"] = run_id
    config["created_at"] = created_at
    paths.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    detection_threshold = float(req.detection_threshold) if req.detection_threshold is not None else load_persisted_threshold(settings, req.model_id)

    total_candidates = 0
    evasive_candidates = 0
    all_records: list[dict[str, Any]] = []
    all_scores: list[float] = []
    all_evasive: list[bool] = []

    for round_idx in range(req.rounds):
        seed_rows = _pick_seed_rows(rows=rows, seeds_per_round=req.seeds_per_round, seed=req.seed + round_idx)
        round_records: list[dict[str, Any]] = []
        round_texts: list[str] = []

        for base in seed_rows:
            base_text = str(base.get("text") or "").strip()
            if not base_text:
                continue

            generated = mutate_text_with_backend(
                settings=settings,
                base_text=base_text,
                num_candidates=req.candidates_per_seed,
                seed=req.seed + round_idx,
                actions=["lexical_swap", "obfuscate", "urgency"],
                similarity_threshold=req.similarity_threshold,
                require_anchors=req.require_anchors,
            )

            for g in generated:
                cand_text = str(g.get("text") or "").strip()
                if not cand_text:
                    continue
                rec = {
                    "round": int(round_idx),
                    "base_id": base.get("id"),
                    "base_label": int(base.get("label", 0)),
                    "candidate_text": cand_text,
                    "generator": g.get("metadata"),
                    "similarity": float(g.get("similarity", 0.0)),
                    "actions": g.get("actions"),
                }
                round_records.append(rec)
                round_texts.append(cand_text)

        scores, evasive = _score_candidates(
            settings=settings,
            model_id=req.model_id,
            texts=round_texts,
            detection_threshold=detection_threshold,
            dry_run=req.dry_run,
            seed=req.seed + 1000 + round_idx,
        )

        for rec, s, ev in zip(round_records, scores, evasive, strict=False):
            rec["scam_probability"] = float(s)
            rec["evasive"] = bool(ev)

        total_candidates += len(round_records)
        evasive_candidates += sum(1 for r in round_records if bool(r.get("evasive")))

        all_records.extend(round_records)
        all_scores.extend([float(s) for s in scores])
        all_evasive.extend([bool(ev) for ev in evasive])

        round_path = paths.root_dir / f"round_{round_idx}.jsonl"
        write_jsonl(round_path, round_records)

    evasion_rate = float(evasive_candidates) / float(total_candidates) if total_candidates else 0.0

    hard = [r for r in all_records if bool(r.get("evasive"))]
    hard.sort(key=lambda r: (float(r.get("scam_probability", 0.0)), -float(r.get("similarity", 0.0))))
    hard = hard[: int(req.hard_max_examples)]

    hard_path = paths.root_dir / "hard_examples.jsonl"
    write_jsonl(hard_path, hard)

    augmented_rows: list[dict[str, Any]] = []
    for r in hard:
        augmented_rows.append(
            {
                "id": str(uuid.uuid4()),
                "text": str(r.get("candidate_text") or "").strip(),
                "label": 1,
                "source": "scamevo_synth",
                "channel": "sms",
                "metadata": {
                    "synthetic": True,
                    "watermark": "SCAMEVO_SYNTH_v1",
                    "created_at": created_at,
                    "run_id": run_id,
                    "base_id": r.get("base_id"),
                    "similarity": r.get("similarity"),
                    "generator": r.get("generator"),
                    "actions": r.get("actions"),
                },
            }
        )

    augmented_dataset_meta = create_augmented_dataset(
        settings=settings,
        base_dataset_id=req.dataset_id,
        added_train_rows=augmented_rows,
        dataset_name=f"augmented_{req.dataset_id}",
        seed=req.seed,
    )

    base_model_meta = load_model_meta(settings, req.model_id)
    base_model_name = str(base_model_meta.get("base_model") or "distilbert-base-uncased")
    retrain_base_model = str(req.base_model or base_model_name)

    train_req = DetectorTrainRequest(
        dataset_id=augmented_dataset_meta["dataset_id"],
        backend=req.retrain_backend,
        base_model=retrain_base_model,
        max_length=int(req.max_length),
        epochs=int(req.epochs),
        batch_size=int(req.batch_size),
        learning_rate=float(req.learning_rate),
        weight_decay=float(req.weight_decay),
        seed=int(req.seed),
        detection_threshold=float(detection_threshold),
        tfidf_ngram_min=int(req.tfidf_ngram_min),
        tfidf_ngram_max=int(req.tfidf_ngram_max),
        tfidf_max_features=int(req.tfidf_max_features),
        tfidf_analyzer=str(req.tfidf_analyzer),
        logreg_c=float(req.logreg_c),
        logreg_class_weight=req.logreg_class_weight,
    )

    defended_model_id: str | None
    defended_eval: dict[str, Any]

    if req.dry_run:
        defended_model_id = str(req.model_id)
        defended_eval = {}
    else:
        defended = train_detector(settings=settings, req=train_req)
        defended_model_id = str(defended.get("model_id"))
        defended_eval = {}

    baseline_eval, _ = evaluate_model_on_dataset(
        settings=settings,
        model_id=req.model_id,
        dataset_id=req.dataset_id,
        split="holdout",
        detection_threshold=detection_threshold,
    )

    if req.dry_run:
        defended_eval = dict(baseline_eval)
    else:
        defended_eval, _ = evaluate_model_on_dataset(
            settings=settings,
            model_id=str(defended_model_id),
            dataset_id=req.dataset_id,
            split="holdout",
            detection_threshold=detection_threshold,
        )

    attacked_metrics = _binary_metrics(
        y_true=[1 for _ in all_scores],
        probs=all_scores,
        detection_threshold=detection_threshold,
    )

    defended_attack_scores, defended_attack_evasive = _score_candidates(
        settings=settings,
        model_id=str(defended_model_id) if not req.dry_run else None,
        texts=[str(r.get("candidate_text") or "") for r in all_records],
        detection_threshold=detection_threshold,
        dry_run=req.dry_run,
        seed=req.seed + 4242,
    )
    defended_attack_evasion_rate = (
        float(sum(1 for ev in defended_attack_evasive if bool(ev))) / float(len(defended_attack_evasive))
        if defended_attack_evasive
        else 0.0
    )

    summary = {
        "run_id": run_id,
        "created_at": created_at,
        "run_type": run_type,
        "dataset_id": req.dataset_id,
        "model_id": req.model_id,
        "defended_model_id": defended_model_id,
        "augmented_dataset_id": augmented_dataset_meta.get("dataset_id"),
        "rounds": int(req.rounds),
        "total_candidates": int(total_candidates),
        "evasive_candidates": int(evasive_candidates),
        "evasion_rate": float(evasion_rate),
        "hard_examples": int(len(hard)),
        "metrics": {
            "baseline_eval": baseline_eval,
            "attacked": attacked_metrics,
            "defended_eval": defended_eval,
            "defended_attack": {
                "num_samples": int(len(defended_attack_scores)),
                "evasion_rate": float(defended_attack_evasion_rate),
                "threshold": float(detection_threshold),
            },
            "delta": {
                "evasion_rate": float(evasion_rate) - float(defended_attack_evasion_rate),
                "defended_f1_minus_baseline_f1": float(defended_eval.get("f1", 0.0)) - float(baseline_eval.get("f1", 0.0)),
            },
        },
    }

    paths.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    insert_run(
        settings=settings,
        run_id=run_id,
        created_at=created_at,
        run_type=run_type,
        config_path=str(paths.config_path),
        artifacts_dir=str(paths.root_dir),
    )

    return {
        "run_id": run_id,
        "created_at": created_at,
        "run_type": run_type,
        "artifacts_dir": str(paths.root_dir),
        "config_path": str(paths.config_path),
        "summary": summary,
    }


def run_adversarial(*, settings: Settings, req: AdversarialRunRequest) -> dict[str, Any]:
    ds = dataset_paths(settings, req.dataset_id)
    path = {"train": ds.train_path, "eval": ds.eval_path, "holdout": ds.holdout_path}[req.split]
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError("Selected dataset split is empty")

    run_id = str(uuid.uuid4())
    created_at = _utc_now_iso()
    run_type = "adversarial"

    paths = run_paths(settings, run_id)
    paths.root_dir.mkdir(parents=True, exist_ok=True)

    config = req.model_dump()
    config["run_id"] = run_id
    config["created_at"] = created_at
    paths.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    total_candidates = 0
    evasive_candidates = 0

    for round_idx in range(req.rounds):
        seed_rows = _pick_seed_rows(rows=rows, seeds_per_round=req.seeds_per_round, seed=req.seed + round_idx)

        round_records: list[dict[str, Any]] = []
        round_texts: list[str] = []

        for base in seed_rows:
            base_text = str(base.get("text") or "").strip()
            if not base_text:
                continue

            generated = mutate_text_with_backend(
                settings=settings,
                base_text=base_text,
                num_candidates=req.candidates_per_seed,
                seed=req.seed + round_idx,
                actions=["lexical_swap", "obfuscate", "urgency"],
                similarity_threshold=req.similarity_threshold,
                require_anchors=req.require_anchors,
            )

            for g in generated:
                rec = {
                    "round": int(round_idx),
                    "base_id": base.get("id"),
                    "base_label": int(base.get("label", 0)),
                    "candidate_text": g.get("text"),
                    "generator": g.get("metadata"),
                    "similarity": float(g.get("similarity", 0.0)),
                    "actions": g.get("actions"),
                }
                round_records.append(rec)
                round_texts.append(str(g.get("text") or ""))

        scores, evasive = _score_candidates(
            settings=settings,
            model_id=req.model_id,
            texts=round_texts,
            detection_threshold=req.detection_threshold,
            dry_run=req.dry_run,
            seed=req.seed + 1000 + round_idx,
        )

        for rec, s, ev in zip(round_records, scores, evasive, strict=False):
            rec["scam_probability"] = float(s)
            rec["evasive"] = bool(ev)

        total_candidates += len(round_records)
        evasive_candidates += sum(1 for r in round_records if bool(r.get("evasive")))

        round_path = paths.root_dir / f"round_{round_idx}.jsonl"
        write_jsonl(round_path, round_records)

    evasion_rate = float(evasive_candidates) / float(total_candidates) if total_candidates else 0.0

    summary = {
        "run_id": run_id,
        "created_at": created_at,
        "run_type": run_type,
        "dataset_id": req.dataset_id,
        "model_id": req.model_id,
        "rounds": int(req.rounds),
        "total_candidates": int(total_candidates),
        "evasive_candidates": int(evasive_candidates),
        "evasion_rate": float(evasion_rate),
    }

    paths.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    insert_run(
        settings=settings,
        run_id=run_id,
        created_at=created_at,
        run_type=run_type,
        config_path=str(paths.config_path),
        artifacts_dir=str(paths.root_dir),
    )

    return {
        "run_id": run_id,
        "created_at": created_at,
        "run_type": run_type,
        "artifacts_dir": str(paths.root_dir),
        "config_path": str(paths.config_path),
        "summary": summary,
    }


def load_run_summary(*, artifacts_dir: str) -> dict[str, Any]:
    p = Path(artifacts_dir) / "summary.json"
    if not p.exists():
        raise FileNotFoundError("Run summary not found")
    return json.loads(p.read_text(encoding="utf-8"))
