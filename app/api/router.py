from fastapi import APIRouter

from app.api.routes import adversarial
from app.api.routes import dataset
from app.api.routes import detector
from app.api.routes import generator
from app.api.routes import health
from app.api.routes import robustness

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(dataset.router, tags=["dataset"])
api_router.include_router(detector.router, tags=["detector"])
api_router.include_router(generator.router, tags=["generator"])
api_router.include_router(adversarial.router, tags=["adversarial"])
api_router.include_router(robustness.router, tags=["robustness"])
