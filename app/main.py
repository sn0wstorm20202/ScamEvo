import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.request_id import RequestIdMiddleware
from app.core.structured_logging import configure_logging
from app.core.paths import ensure_storage_layout
from app.db.metadata import init_db

def create_app() -> FastAPI:
    settings = get_settings()

    configure_logging()

    app = FastAPI(title=settings.app_name)

    app.add_middleware(RequestIdMiddleware)

    raw_origins = os.getenv("SCAMEVO_CORS_ORIGINS", "").strip()
    if raw_origins:
        allow_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    else:
        allow_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]

    allow_credentials = (
        os.getenv("SCAMEVO_CORS_ALLOW_CREDENTIALS", "0").strip().lower() in {"1", "true", "yes", "y"}
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        ensure_storage_layout(settings)
        init_db(settings)

    app.include_router(api_router)

    return app


app = create_app()
