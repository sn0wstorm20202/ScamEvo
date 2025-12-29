from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.demo import DEMO_DETECTOR_EVALUATE, DEMO_DETECTOR_TRAIN
from app.schemas.detector import (
    DetectorEvaluateResponse,
    DetectorInferRequest,
    DetectorInferResponse,
    DetectorTrainRequest,
    DetectorTrainResponse,
)
from app.services.detector import evaluate_model_on_dataset, infer_detector, train_detector

router = APIRouter(prefix="/detector")


@router.post("/train", response_model=DetectorTrainResponse)
def train(req: DetectorTrainRequest):
    settings = get_settings()
    if settings.demo_mode:
        req = req.model_copy(update={k: v for k, v in DEMO_DETECTOR_TRAIN.items()})
    try:
        result = train_detector(settings=settings, req=req)
        return DetectorTrainResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@router.post("/infer", response_model=DetectorInferResponse)
def infer(req: DetectorInferRequest):
    settings = get_settings()
    try:
        items = infer_detector(settings=settings, model_id=req.model_id, texts=req.texts, explain=req.explain)
        return DetectorInferResponse(model_id=req.model_id, items=items)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@router.get("/evaluate", response_model=DetectorEvaluateResponse)
def evaluate(model_id: str, dataset_id: str, split: str = "holdout", detection_threshold: float = 0.5):
    settings = get_settings()

    if settings.demo_mode:
        split = str(DEMO_DETECTOR_EVALUATE.get("split"))
        detection_threshold = float(DEMO_DETECTOR_EVALUATE.get("detection_threshold"))

    if split not in {"train", "eval", "holdout"}:
        raise HTTPException(status_code=400, detail="split must be one of train|eval|holdout")

    try:
        metrics, false_negatives = evaluate_model_on_dataset(
            settings=settings,
            model_id=model_id,
            dataset_id=dataset_id,
            split=split,
            detection_threshold=detection_threshold,
        )
        return DetectorEvaluateResponse(
            model_id=model_id,
            dataset_id=dataset_id,
            split=split,
            detection_threshold=detection_threshold,
            metrics=metrics,
            false_negatives=false_negatives,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
