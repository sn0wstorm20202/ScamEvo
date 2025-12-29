from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.demo import DEMO_DATASET_OPTIONS
from app.schemas.dataset import DatasetSampleResponse, DatasetSummary, DatasetUploadOptions
from app.services.datasets import dataset_paths, ingest_dataset, load_dataset_meta
from app.services.jsonl import read_jsonl

router = APIRouter(prefix="/dataset")


@router.post("/upload", response_model=DatasetSummary)
async def upload_dataset(
    file: UploadFile = File(...),
    options: str = Form("{}"),
):
    settings = get_settings()

    try:
        opts = DatasetUploadOptions(**json.loads(options))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid options JSON: {e}")

    if settings.demo_mode:
        opts = opts.model_copy(update={k: v for k, v in DEMO_DATASET_OPTIONS.items()})

    try:
        file_bytes = await file.read()
        meta = ingest_dataset(
            settings=settings,
            original_filename=file.filename or "uploaded_dataset",
            file_bytes=file_bytes,
            opts=opts,
        )
        return DatasetSummary(**meta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest dataset: {e}")


@router.get("/summary", response_model=DatasetSummary)
def dataset_summary(dataset_id: str):
    settings = get_settings()
    try:
        meta = load_dataset_meta(settings, dataset_id)
        return DatasetSummary(**meta)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/sample", response_model=DatasetSampleResponse)
def dataset_sample(
    dataset_id: str,
    split: str = "train",
    n: int = 5,
    label: Optional[int] = None,
):
    settings = get_settings()
    meta = load_dataset_meta(settings, dataset_id)
    paths = dataset_paths(settings, dataset_id)

    if split not in {"train", "eval", "holdout"}:
        raise HTTPException(status_code=400, detail="split must be one of train|eval|holdout")

    path = {"train": paths.train_path, "eval": paths.eval_path, "holdout": paths.holdout_path}[split]

    samples = []
    for row in read_jsonl(path):
        if label is not None and int(row.get("label")) != int(label):
            continue
        samples.append(row)
        if len(samples) >= n:
            break

    return DatasetSampleResponse(dataset_id=dataset_id, split=split, samples=samples)
