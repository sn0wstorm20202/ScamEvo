from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter()


@router.get("/health")
def health():
    s = get_settings()
    return {
        "app": s.app_name,
        "research_mode": s.research_mode,
        "do_not_deploy": s.do_not_deploy,
    }
