# SCAM-EVO Backend — Project Overview

## Goal

Build a controlled adversarial ML engine that:

- Trains a scam detector (defender)
- Generates semantic-preserving scam variants (attacker)
- Co-evolves defender vs attacker over multiple rounds
- Measures robustness and degradation curves

Backend is the “brain”. UI is a client.

## Design principles

- Deterministic runs
  - Every training run stores its config and seed.
  - Artifacts are versioned in filesystem folders.
- Minimal infrastructure
  - Heavy artifacts (datasets/models/logs) are stored in `storage/`.
  - DB stores only metadata.
- Safety
  - Research-mode gating is supported via env vars.
  - Generator/export features will be restricted in MVP.

## Architecture

### 1) API layer (FastAPI)

- Router entry: `app/main.py` → `app/api/router.py`
- Route modules: `app/api/routes/*`

### 2) Services layer

- `app/services/datasets.py`
  - Ingest + normalize + split datasets
  - Produces canonical JSONL and a `meta.json`
- `app/services/detector.py`
  - Train detector backends:
    - TF-IDF + Logistic Regression (`tfidf_logreg`)
    - HuggingFace transformer classifier (`hf_transformer`, optional)
  - Save model artifacts (sklearn joblib or HF model/tokenizer)
  - Inference with probability outputs
  - Evaluation metrics + false negatives export
 - `app/services/generator.py`
  - Controlled mutation actions (research-gated)
 - `app/services/adversarial.py`
  - Round-based mutation + scoring loop (research-gated)

### 3) Storage

- `storage/raw_datasets/` original uploads
- `storage/datasets/` normalized + split JSONL
- `storage/models/` versioned model artifacts
- `storage/runs/` adversarial run artifacts

Training scripts (repo root):

- `scripts/train_unified_model.py` build a unified dataset from multiple sources and train a baseline model
- `scripts/train_best_model.py` offline best-model search (TF-IDF variants + threshold tuning)
- `scripts/train_best_model_cv.py` GridSearchCV-based best-model search (cross-validated hyperparameter tuning)

### 4) Metadata DB

- Adapter: `app/db/metadata.py`
- Backends:
  - Postgres (Neon) when `SCAMEVO_DATABASE_URL` is set
  - SQLite fallback otherwise

Tables:

- `datasets`
- `models`
- `runs`

## Canonical dataset schema (internal)

Each stored sample is normalized into JSONL rows:

- `id` (uuid)
- `text` (string)
- `label` (0=legit, 1=scam)
- `source` (public_dataset | synthetic)
- `channel` (sms)
- `metadata` (free-form dict)

## Endpoints implemented

### Health

- `GET /health`

### Dataset

- `POST /dataset/upload`
- `GET /dataset/summary?dataset_id=...`
- `GET /dataset/sample?dataset_id=...&split=train|eval|holdout&n=...`

### Detector

- `POST /detector/train`
- `POST /detector/infer`
- `GET /detector/evaluate?model_id=...&dataset_id=...&split=...`

### Generator (research-gated)

- `POST /generator/mutate`

### Adversarial (research-gated)

- `POST /adversarial/run`
- `POST /adversarial/retrain`
- `GET /adversarial/history`

### Robustness

- `GET /robustness/report?run_id=...`

## What’s planned next (PRD alignment)

- Adversarial retraining loop
  - Implemented via `POST /adversarial/retrain` (generate attacks → select hard examples → augment dataset → retrain defender → evaluate deltas)
- Robustness curves
  - Accuracy vs mutation depth
  - Confidence decay curves
  - FN rate on unseen mutations
- Explainability improvements
  - Stable span highlighting
  - Better token normalization

## Current operational notes

- Generator backend selection:
  - `SCAMEVO_GENERATOR_BACKEND=rule` (default) for fully offline demos
  - `SCAMEVO_GENERATOR_BACKEND=llm` (optional) requires `SCAMEVO_OPENAI_API_KEY`
- Threshold consistency:
  - `detection_threshold` is persisted per model in `storage/models/<model_id>/train_config.json` and used by default.
- Structured logging:
  - JSON logs with `X-Request-ID` propagation for correlation across requests.
