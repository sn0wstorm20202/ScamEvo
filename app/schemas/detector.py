from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class DetectorTrainRequest(BaseModel):
    dataset_id: str

    backend: Literal["hf_transformer", "tfidf_logreg"] = "hf_transformer"

    base_model: str = "distilbert-base-uncased"
    max_length: int = 192

    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01

    seed: int = 1337

    detection_threshold: float = 0.5

    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    tfidf_max_features: int = 50000

    tfidf_analyzer: Literal["word", "char_wb"] = "word"

    logreg_c: float = 1.0
    logreg_class_weight: Optional[Literal["balanced"]] = None


class DetectorTrainResponse(BaseModel):
    model_id: str
    version: str
    created_at: str
    dataset_id: str

    base_model: str

    metrics: dict[str, Any]


class DetectorInferRequest(BaseModel):
    model_id: str
    texts: list[str] = Field(min_length=1)
    explain: bool = False
    detection_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class TokenImportance(BaseModel):
    token: str
    score: float


class DetectorInferItem(BaseModel):
    text: str
    scam_probability: float
    prediction: Literal[0, 1]
    token_importance: Optional[list[TokenImportance]] = None


class DetectorInferResponse(BaseModel):
    model_id: str
    items: list[DetectorInferItem]


class DetectorEvaluateResponse(BaseModel):
    model_id: str
    dataset_id: str
    split: Literal["train", "eval", "holdout"]

    detection_threshold: float
    metrics: dict[str, Any]
    false_negatives: list[dict[str, Any]]
