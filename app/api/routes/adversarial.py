from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.demo import DEMO_ADVERSARIAL_RUN
from app.db.metadata import get_run, list_runs
from app.schemas.adversarial import (
    AdversarialHistoryResponse,
    AdversarialRoundRecordsResponse,
    AdversarialRunDetailResponse,
    AdversarialRunRequest,
    AdversarialRunResponse,
    AdversarialRoundStats,
    RunRecord,
)
from app.services.adversarial import load_round_records, load_run_summary, run_adversarial

router = APIRouter(prefix="/adversarial")


@router.post("/run", response_model=AdversarialRunResponse)
def run(req: AdversarialRunRequest):
    settings = get_settings()
    if not settings.research_mode or not settings.do_not_deploy:
        raise HTTPException(status_code=403, detail="Adversarial engine is disabled")

    if settings.demo_mode:
        req = req.model_copy(update={k: v for k, v in DEMO_ADVERSARIAL_RUN.items()})

    try:
        out = run_adversarial(settings=settings, req=req)
        return AdversarialRunResponse(**out)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run adversarial simulation: {e}")


@router.get("/history", response_model=AdversarialHistoryResponse)
def history(limit: int = 50):
    settings = get_settings()
    rows = list_runs(settings=settings, limit=limit)
    return AdversarialHistoryResponse(
        runs=[RunRecord(**r) for r in rows],
        metadata={"limit": int(limit), "count": len(rows)},
    )


@router.get("/detail", response_model=AdversarialRunDetailResponse)
def detail(run_id: str):
    settings = get_settings()
    rec = get_run(settings=settings, run_id=run_id)
    if rec is None or not rec.get("artifacts_dir"):
        raise HTTPException(status_code=404, detail=f"Unknown run_id={run_id}")

    summary = load_run_summary(artifacts_dir=str(rec["artifacts_dir"]))
    rounds = int(summary.get("rounds") or 0)

    per_round: list[AdversarialRoundStats] = []
    for r in range(rounds):
        rows = load_round_records(settings=settings, run_id=run_id, round_idx=r, limit=100000)
        total = int(len(rows))
        evasive = int(sum(1 for x in rows if bool(x.get("evasive"))))
        rate = float(evasive) / float(total) if total else 0.0
        per_round.append(
            AdversarialRoundStats(
                round=int(r),
                total_candidates=total,
                evasive_candidates=evasive,
                evasion_rate=float(rate),
            )
        )

    return AdversarialRunDetailResponse(
        run_id=str(summary.get("run_id")),
        created_at=str(summary.get("created_at")),
        run_type=str(summary.get("run_type")),
        dataset_id=str(summary.get("dataset_id")),
        model_id=summary.get("model_id"),
        rounds=rounds,
        per_round=per_round,
    )


@router.get("/round", response_model=AdversarialRoundRecordsResponse)
def round_records(run_id: str, round: int = 0, limit: int = 2000):
    settings = get_settings()
    rec = get_run(settings=settings, run_id=run_id)
    if rec is None or not rec.get("artifacts_dir"):
        raise HTTPException(status_code=404, detail=f"Unknown run_id={run_id}")

    try:
        rows = load_round_records(settings=settings, run_id=run_id, round_idx=int(round), limit=int(limit))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return AdversarialRoundRecordsResponse(
        run_id=run_id,
        round=int(round),
        records=rows,
        metadata={"limit": int(limit), "count": len(rows)},
    )
