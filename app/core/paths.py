from __future__ import annotations

from pathlib import Path

from app.core.config import Settings


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_storage_layout(settings: Settings) -> None:
    ensure_dir(settings.storage_dir)
    ensure_dir(settings.raw_datasets_dir)
    ensure_dir(settings.datasets_dir)
    ensure_dir(settings.models_dir)
    ensure_dir(settings.runs_dir)
