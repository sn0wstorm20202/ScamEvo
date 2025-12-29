from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.db.metadata import get_run
from app.schemas.robustness import RobustnessReport
from app.services.adversarial import load_run_summary

router = APIRouter(prefix="/robustness")


@router.get("/report", response_model=RobustnessReport)
def report(run_id: str):
    settings = get_settings()
    rec = get_run(settings=settings, run_id=run_id)
    if rec is None or not rec.get("artifacts_dir"):
        raise HTTPException(status_code=404, detail=f"Unknown run_id={run_id}")

    summary = load_run_summary(artifacts_dir=str(rec["artifacts_dir"]))

    return RobustnessReport(
        run_id=summary.get("run_id"),
        created_at=summary.get("created_at"),
        run_type=summary.get("run_type"),
        metrics={
            "dataset_id": summary.get("dataset_id"),
            "model_id": summary.get("model_id"),
            "rounds": summary.get("rounds"),
            "total_candidates": summary.get("total_candidates"),
            "evasive_candidates": summary.get("evasive_candidates"),
            "evasion_rate": summary.get("evasion_rate"),
        },
    )
