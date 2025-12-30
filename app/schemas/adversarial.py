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


class AdversarialRetrainRequest(BaseModel):
    dataset_id: str
    model_id: str
    split: Literal["train", "eval", "holdout"] = "train"

    rounds: int = Field(default=1, ge=1, le=50)
    seeds_per_round: int = Field(default=25, ge=1, le=500)
    candidates_per_seed: int = Field(default=5, ge=1, le=50)

    detection_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    require_anchors: bool = True

    hard_max_examples: int = Field(default=250, ge=1, le=5000)
    seed: int = 1337
    dry_run: bool = False

    retrain_backend: Literal["tfidf_logreg", "hf_transformer"] = "tfidf_logreg"

    base_model: Optional[str] = None
    max_length: int = 192
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01

    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    tfidf_max_features: int = 50000
    tfidf_analyzer: Literal["word", "char_wb"] = "word"
    logreg_c: float = 1.0
    logreg_class_weight: Optional[Literal["balanced"]] = None


class AdversarialRetrainResponse(BaseModel):
    run_id: str
    created_at: str
    run_type: str
    artifacts_dir: str
    config_path: str
    summary: dict[str, Any]
