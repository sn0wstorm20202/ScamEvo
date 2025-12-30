from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.schemas.generator import GeneratorMutateRequest, GeneratorMutateResponse
from app.services.generator import mutate_text_with_backend

router = APIRouter(prefix="/generator")


@router.post("/mutate", response_model=GeneratorMutateResponse)
def mutate(req: GeneratorMutateRequest):
    settings = get_settings()
    if not settings.research_mode or not settings.do_not_deploy:
        raise HTTPException(status_code=403, detail="Generator is disabled")

    candidates = mutate_text_with_backend(
        settings=settings,
        base_text=req.text,
        num_candidates=req.num_candidates,
        seed=req.seed,
        actions=req.actions,
        similarity_threshold=req.similarity_threshold,
        require_anchors=req.require_anchors,
    )

    return GeneratorMutateResponse(base_text=req.text, candidates=candidates)
