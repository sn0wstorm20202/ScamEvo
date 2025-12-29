from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class DatasetUploadOptions(BaseModel):
    dataset_name: Optional[str] = None
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    scam_label_values: list[str] = Field(default_factory=lambda: ["spam", "smish", "smishing", "scam", "fraud", "phishing", "1", "true"])
    legit_label_values: list[str] = Field(default_factory=lambda: ["ham", "legit", "0", "false"])

    default_label: Optional[int] = None
    channel: str = "sms"

    train_ratio: float = 0.8
    eval_ratio: float = 0.1
    holdout_ratio: float = 0.1

    seed: int = 1337
    balance_strategy: Literal["none", "downsample_majority"] = "none"


class DatasetSummary(BaseModel):
    dataset_id: str
    version: str
    created_at: str
    source_filename: str

    num_samples: int
    num_scam: int
    num_legit: int

    splits: dict[str, int]
    label_mapping: dict[str, Any]


class DatasetSampleResponse(BaseModel):
    dataset_id: str
    split: Literal["train", "eval", "holdout"]
    samples: list[dict[str, Any]]
