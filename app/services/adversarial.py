from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.db.metadata import insert_run
from app.schemas.adversarial import AdversarialRunRequest
from app.services.datasets import dataset_paths
from app.services.detector import infer_detector
from app.services.generator import mutate_text
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

            generated = mutate_text(
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
