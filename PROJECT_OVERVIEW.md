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
  - Train transformer classifier
  - Save HF model/tokenizer
  - Inference with probability outputs
  - Evaluation metrics + false negatives export

### 3) Storage

- `storage/raw_datasets/` original uploads
- `storage/datasets/` normalized + split JSONL
- `storage/models/` versioned model artifacts
- `storage/runs/` reserved for adversarial runs (planned)

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

## What’s planned next (PRD alignment)

- Generator module (`/generator/mutate`)
  - Controlled mutation actions
  - Semantic similarity filter (Sentence-BERT)
  - Scam anchor rules + watermarking
- Adversarial training engine (`/adversarial/run`, `/adversarial/history`)
  - Round-based sampling → mutation → scoring → retraining
  - Full reproducibility (configs + seeds + logs)
- Robustness module (`/robustness/report`)
  - Accuracy vs mutation depth
  - Confidence decay curves
  - FN rate on unseen mutations
- Explainability module (`/explain/{sample_id}`)
  - Token importance, highlighting
