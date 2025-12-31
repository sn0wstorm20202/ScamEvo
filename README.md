# SCAM-EVO Backend (FastAPI)

Backend-only **FastAPI** service for a controlled adversarial ML system: ingest datasets, train a scam detector, simulate scam message evolution (attacker), and generate run artifacts + robustness summaries.

This MVP uses **supervised** learning (binary classification: legit vs scam). **No unsupervised training** is used in the current implementation.

Research-gated features:

- Set `SCAMEVO_RESEARCH_MODE=1` and `SCAMEVO_DO_NOT_DEPLOY=1` to enable generator/adversarial endpoints in a controlled environment.

## What’s implemented (current)

- Health
  - `GET /health`
- Dataset module
  - `POST /dataset/upload`
  - `GET /dataset/summary`
  - `GET /dataset/sample`
  - Normalizes multiple dataset formats into a canonical JSONL schema and writes split files.
- Detector module
  - `POST /detector/train`
  - `POST /detector/infer`
  - `GET /detector/evaluate`
  - Backends:
    - `tfidf_logreg` (TF-IDF + Logistic Regression) — fast/offline baseline
    - `hf_transformer` (HuggingFace Transformers) — optional heavier baseline
  - Saves model artifacts + metrics to filesystem and metadata to DB.
- Generator module (research-gated)
  - `POST /generator/mutate`
  - Controlled mutations with similarity filtering and optional anchor checks.
  - Backends:
    - `rule` (default) — fully offline
    - `llm` (OpenAI) — optional, requires `SCAMEVO_OPENAI_API_KEY`
  - Recommended for demos: set `SCAMEVO_GENERATOR_BACKEND=rule`.
- Adversarial module (research-gated)
  - `POST /adversarial/run`
  - `POST /adversarial/retrain`
  - `GET /adversarial/history`
  - Round-based mutation + scoring loop, logs JSONL per round and a run summary.
- Robustness module
  - `GET /robustness/report`
  - Produces a report from run summary.
    - For `adversarial`: evasion rate + counts.
    - For `adversarial_retrain`: baseline vs attacked vs defended metrics + deltas.
- Inference threshold consistency
  - Detection threshold is persisted per model in `storage/models/<model_id>/train_config.json`.
  - `POST /detector/infer` and `GET /detector/evaluate` default to the persisted threshold unless you override.
- Structured logging + correlation IDs
  - JSON logs.
  - `X-Request-ID` is accepted/returned and attached to logs.
- Metadata DB
  - Uses **Neon Postgres** when `SCAMEVO_DATABASE_URL` is set.
  - Falls back to **SQLite** when not set (useful for local dev/tests).

## Tech stack

- Python
- FastAPI
- scikit-learn (TF-IDF + Logistic Regression)
- PyTorch + HuggingFace Transformers (optional detector backend)
- Metadata DB: Neon Postgres (optional) / SQLite fallback
- Artifact storage: filesystem (`storage/`)

## Project layout

- `app/main.py` FastAPI entrypoint
- `app/api/routes/` HTTP routes
- `app/services/` core logic
- `app/db/` DB adapters
- `scripts/run_demo_story.py` reproducible end-to-end demo runner
- `scripts/train_unified_model.py` build a unified dataset from multiple sources and train a single model
- `scripts/train_best_model.py` offline best-model search (TF-IDF variants + threshold tuning)
- `scripts/train_best_model_cv.py` GridSearchCV-based best-model search (cross-validated TF-IDF + Logistic Regression)
- `storage/` created at runtime (datasets, models, runs)

## Setup (Windows)

### 1) Create venv

```powershell
python -m venv .venv
```

### 2) Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3) Configure environment

Recommended: copy `.env.example` to `.env` and edit values.

#### Option A (recommended): Neon Postgres for metadata

Set:

- `SCAMEVO_DATABASE_URL` = your Neon connection string

Example (PowerShell):

```powershell
$env:SCAMEVO_DATABASE_URL = "postgresql://USER:PASSWORD@HOST/DB?sslmode=require"
```

#### OpenAI key (only if using LLM generator)

SCAM-EVO reads:

- `SCAMEVO_OPENAI_API_KEY`

Example (PowerShell):

```powershell
$env:SCAMEVO_OPENAI_API_KEY = "..."
```

Note: if you set env vars in the Windows Environment Variables UI, restart your terminal/IDE so the running process inherits them.

#### Option B: SQLite fallback (no remote DB)

No config required.

Optional:

- `SCAMEVO_STORAGE_DIR` (default: `<repo>/storage`)
- `SCAMEVO_DB_PATH` (default: `<storage>/metadata.sqlite3`)

### 4) Run the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Health check:

- `GET http://127.0.0.1:8000/health`

OpenAPI:

- `http://127.0.0.1:8000/docs`

## Quickstart (local, reproducible)

Run an end-to-end demo flow (dataset upload -> train -> evaluate -> adversarial run -> adversarial retrain -> robustness reports).

```powershell
.\.venv\Scripts\python.exe scripts\run_demo_story.py
```

Outputs:

- `storage/demo/<retrain_run_id>/experiment_manifest.json`
- `storage/demo/<retrain_run_id>/robustness_report.json`
- `storage/demo/<retrain_run_id>/summary.md`
- `storage/demo/<retrain_run_id>/robustness_attack_report.json`

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Train a unified (global) model from multiple datasets

This repo includes scripts to train a single model across multiple files under `datasets/`.

Build a unified dataset and train a baseline model:

```powershell
.\.venv\Scripts\python.exe scripts\train_unified_model.py --profile sms --force-sqlite
```

Outputs:

- `storage/unified/unified_sms_manifest.json` (sources + unified `dataset_id` + resulting `model_id`)
- `storage/models/<model_id>/...` (trained model artifact)

Baseline unified model snapshot (from `storage/unified/unified_sms_manifest.json`):

- `model_id`: `a340550d-afaa-4fe4-a46f-9b1cb655c421`
- `unified dataset_id`: `042bfd69-580a-472b-832b-19a29da10410`
- Eval metrics (threshold=0.5):
  - accuracy: `0.9655737704918033`
  - precision: `0.9885057471264368`
  - recall: `0.8113207547169812`
  - f1: `0.8911917098445595`

## Best model search (offline)

This runs an offline hyperparameter search (word/character TF-IDF n-grams + class weighting) and optionally tunes the classification threshold using the eval split.

```powershell
.\.venv\Scripts\python.exe scripts\train_best_model.py --profile sms --force-sqlite --objective f1 --optimize-threshold
```

Outputs:

- `storage/best/best_sms_manifest.json` (all candidates + selected best)

Current best (from `storage/best/best_sms_manifest.json`):

- `model_id`: `aba3b6f5-f5d8-45c3-aab6-71569539043e`
- `unified dataset_id`: `16c37acb-8151-4dfd-921b-91b191c3d523`
- Holdout metrics (threshold chosen on eval):
  - accuracy: `0.983633387888707`
  - precision: `0.9619047619047619`
  - recall: `0.9439252336448598`
  - f1: `0.9528301886792453`
  - threshold: `0.36`

## Best model search (GridSearchCV + cross-validation)

This runs a `GridSearchCV` over TF-IDF + Logistic Regression hyperparameters using cross-validation on the training split, then (optionally) tunes the decision threshold on the eval split and evaluates once on holdout.

```powershell
.\.venv\Scripts\python.exe scripts\train_best_model_cv.py --profile sms --force-sqlite --refit f1 --cv-folds 5 --optimize-threshold
```

Outputs:

- `storage/best/best_cv_sms_manifest.json`

Best CV run (from `storage/best/best_cv_sms_manifest.json`):

- `model_id`: `779e95d6-4e63-4651-88fe-b81b103e9f01`
- `unified dataset_id`: `3f02c1ec-e838-4c84-946e-a3d1195e71d9`
- Cross-validated score (refit=`f1`): `0.9673964043774816`
- Best params:
  - `tfidf__analyzer`: `char_wb`
  - `tfidf__ngram_range`: `(1, 3)`
  - `tfidf__max_features`: `50000`
  - `clf__C`: `2.0`
  - `clf__class_weight`: `balanced`
- Holdout metrics (threshold chosen on eval):
  - accuracy: `0.9819967266775778`
  - precision: `0.9285714285714286`
  - recall: `0.9719626168224299`
  - f1: `0.9497716894977168`
  - threshold: `0.31`

## Where to find the latest results

All training/evaluation runs write machine-readable outputs under `storage/`:

- `storage/unified/unified_<profile>_manifest.json` (unified dataset build + baseline model)
- `storage/best/best_<profile>_manifest.json` (offline best-search + per-candidate metrics)
- `storage/best/best_cv_<profile>_manifest.json` (GridSearchCV best params + metrics)
- `storage/models/<model_id>/metrics.json` (per-model eval metrics)
- `storage/demo/<retrain_run_id>/experiment_manifest.json` (demo runner output)
- `storage/demo/<retrain_run_id>/robustness_report.json` (defended run)
- `storage/demo/<retrain_run_id>/summary.md` (demo runner output)
- `storage/demo/<retrain_run_id>/robustness_attack_report.json` (attack-only run)

## Storage & artifacts

At runtime the backend creates:

- `storage/raw_datasets/<dataset_id>/...` original upload
- `storage/datasets/<dataset_id>/train.jsonl`
- `storage/datasets/<dataset_id>/eval.jsonl`
- `storage/datasets/<dataset_id>/holdout.jsonl`
- `storage/unified/unified_<profile>_manifest.json`
- `storage/best/best_<profile>_manifest.json`
- `storage/best/best_cv_<profile>_manifest.json`
- `storage/models/<model_id>/sk_model.joblib` (when using `tfidf_logreg`)
- `storage/models/<model_id>/hf_model/` + `tokenizer/` (when using `hf_transformer`)
- `storage/models/<model_id>/metrics.json`
- `storage/models/<model_id>/false_negatives.json`
- `storage/runs/<run_id>/summary.json`
- `storage/runs/<run_id>/round_*.jsonl`
- `storage/runs/<run_id>/hard_examples.jsonl` (for retraining runs)
- `storage/demo/<retrain_run_id>/experiment_manifest.json` (written by demo runner)
- `storage/demo/<retrain_run_id>/robustness_report.json` (defended run)
- `storage/demo/<retrain_run_id>/robustness_attack_report.json` (attack-only run)
- `storage/demo/<retrain_run_id>/summary.md`

## Storage & Database Links

- **Storage artifacts and datasets**: https://drive.google.com/drive/folders/1JisyuUsd-seM8MDHisa4gDIHEtnhjnmf
- **Database exports / backups**: https://drive.google.com/drive/folders/1JisyuUsd-seM8MDHisa4gDIHEtnhjnmf

## API usage quickstart

### Health

```bash
curl "http://127.0.0.1:8000/health"
```

### Upload dataset

Multipart form:

- `file`: dataset file (`.csv` / `.json` / `.jsonl` / `SMSSpamCollection`)
- `options`: JSON string (see `DatasetUploadOptions`)

Example using curl:

```bash
curl -X POST "http://127.0.0.1:8000/dataset/upload" \
  -F "file=@datasets/SMSSpamCollection" \
  -F "options={\"seed\":1337,\"train_ratio\":0.8,\"eval_ratio\":0.1,\"holdout_ratio\":0.1}"
```

### Train detector

#### Train using `tfidf_logreg` (recommended for local/demo)

```bash
curl -X POST "http://127.0.0.1:8000/detector/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "<dataset_id>",
    "backend": "tfidf_logreg",
    "seed": 1337,
    "detection_threshold": 0.5
  }'
```

#### Train using `hf_transformer` (optional)

```bash
curl -X POST "http://127.0.0.1:8000/detector/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "<dataset_id>",
    "backend": "hf_transformer",
    "base_model": "distilbert-base-uncased",
    "epochs": 3,
    "batch_size": 16,
    "seed": 1337
  }'
```

### Inference

```bash
curl -X POST "http://127.0.0.1:8000/detector/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "<model_id>",
    "texts": ["Urgent: verify your UPI to avoid block"],
    "explain": true
  }'
```

### Evaluate

```bash
curl "http://127.0.0.1:8000/detector/evaluate?model_id=<model_id>&dataset_id=<dataset_id>&split=holdout&detection_threshold=0.5"
```

## Notes

- Do **not** commit your Neon connection string to git.
- Rotate the Neon password if it has been shared publicly.
