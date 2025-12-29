from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.demo import DEMO_ADVERSARIAL_RUN
from app.db.metadata import list_runs
from app.schemas.adversarial import AdversarialHistoryResponse, AdversarialRunRequest, AdversarialRunResponse, RunRecord
from app.services.adversarial import run_adversarial

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
