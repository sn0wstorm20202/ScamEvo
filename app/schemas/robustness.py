from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RobustnessReport(BaseModel):
    run_id: str
    created_at: str
    run_type: str
    metrics: dict[str, Any]
