from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class AdversarialRunRequest(BaseModel):
    dataset_id: str
    split: Literal["train", "eval", "holdout"] = "holdout"

    model_id: Optional[str] = None

    rounds: int = Field(default=3, ge=1, le=50)
    seeds_per_round: int = Field(default=25, ge=1, le=500)
    candidates_per_seed: int = Field(default=5, ge=1, le=50)

    detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    require_anchors: bool = True

    seed: int = 1337
    dry_run: bool = False


class AdversarialRunSummary(BaseModel):
    run_id: str
    created_at: str
    run_type: str
    dataset_id: str
    model_id: Optional[str]

    rounds: int
    total_candidates: int
    evasive_candidates: int
    evasion_rate: float


class AdversarialRunResponse(BaseModel):
    run_id: str
    created_at: str
    run_type: str
    artifacts_dir: str
    config_path: str
    summary: AdversarialRunSummary


class RunRecord(BaseModel):
    id: str
    created_at: str
    run_type: str
    config_path: Optional[str] = None
    artifacts_dir: Optional[str] = None


class AdversarialHistoryResponse(BaseModel):
    runs: list[RunRecord]
    metadata: dict[str, Any] = Field(default_factory=dict)
