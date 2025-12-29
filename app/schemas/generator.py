from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


GeneratorAction = Literal["lexical_swap", "obfuscate", "urgency"]


class GeneratorMutateRequest(BaseModel):
    text: str
    num_candidates: int = Field(default=5, ge=1, le=50)
    seed: int = 1337

    actions: list[GeneratorAction] = Field(default_factory=lambda: ["lexical_swap", "obfuscate", "urgency"])

    similarity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    require_anchors: bool = True


class GeneratorCandidate(BaseModel):
    text: str
    similarity: float
    actions: list[GeneratorAction]
    metadata: dict[str, Any]


class GeneratorMutateResponse(BaseModel):
    base_text: str
    candidates: list[GeneratorCandidate]
