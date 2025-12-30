# SCAM-EVO Backend (FastAPI)

Backend-only **FastAPI** service for a controlled adversarial ML system: ingest datasets, train a scam detector, simulate scam message evolution (attacker), and generate run artifacts + robustness summaries.

This MVP uses **supervised** learning (binary classification: legit vs scam). **No unsupervised training** is used in the current implementation.

## What’s implemented (current)

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
  - Controlled, rule-based mutations with similarity filtering and optional anchor checks.
- Adversarial module (research-gated)
  - `POST /adversarial/run`
  - `GET /adversarial/history`
  - Round-based mutation + scoring loop, logs JSONL per round and a run summary.
- Robustness module
  - `GET /robustness/report`
  - Produces a compact report from run summary (evasion rate, counts).
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

#### Option B: SQLite fallback (no remote DB)

No config required.

Optional:

- `SCAMEVO_STORAGE_DIR` (default: `<repo>/storage`)
- `SCAMEVO_DB_PATH` (default: `<storage>/metadata.sqlite3`)

### 4) Run the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

## CORS (Frontend integration)

The backend uses env-configured CORS:

- `SCAMEVO_CORS_ORIGINS` (comma-separated list of allowed origins)
- `SCAMEVO_CORS_ALLOW_CREDENTIALS` (`1/true` to allow credentials; default `0/false`)

For local dev with the Vite frontend:

- `SCAMEVO_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`

Health check:

- `GET http://127.0.0.1:8000/health`

OpenAPI:

- `http://127.0.0.1:8000/docs`

## Connect Backend ↔ Frontend (Dev)

### 1) Backend `.env`

Recommended backend env values for local dev:

- `SCAMEVO_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`
- `SCAMEVO_CORS_ALLOW_CREDENTIALS=0`

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

### 2) Frontend `.env`

In the frontend repo (`Scam-Evo`), set:

- `VITE_API_BASE_URL=http://127.0.0.1:8000`
- `VITE_API_WITH_CREDENTIALS=false`

Start the frontend:

```bash
npm install
npm run dev
```

Open:

- `http://localhost:5173`

Health check:

- `GET http://127.0.0.1:8000/health`

OpenAPI:

- `http://127.0.0.1:8000/docs`

## Testing

Backend unit/integration tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Note: the test suite forces SQLite and ignores `SCAMEVO_DATABASE_URL` to avoid depending on developer Postgres credentials.

## Manual Testing (API + UI)

### Quick API sanity checks

- `GET /health`
- `POST /dataset/upload` (multipart: `file` + `options`)
- `POST /detector/train`
- `GET /detector/evaluate`
- `POST /adversarial/run` (requires `SCAMEVO_RESEARCH_MODE=1` and `SCAMEVO_DO_NOT_DEPLOY=1`)
- `GET /robustness/report`

### End-to-end UI checklist

- **Dataset page**
  - Upload a dataset `.json` file
  - Confirm the table populates from backend sample rows
- **Detector page**
  - Confirm model trains automatically (tfidf baseline) and metrics render
  - Confirm false negatives section shows real backend results
- **Evolution page**
  - Run adversarial simulation and verify timeline populates
  - Click a round and verify the mutation diff updates
- **Robustness page**
  - Confirm robustness report loads (chart + run summary)

## Model testing (end-to-end demo)

Reproducible one-command flow (dataset upload -> train -> evaluate -> adversarial run -> robustness report):

```powershell
.\.venv\Scripts\python.exe scripts\run_demo_story.py --dataset "Dataset_5971.csv" --dataset-name "dataset_5971_demo" --force-sqlite
```

## Storage & artifacts

At runtime the backend creates:

- `storage/raw_datasets/<dataset_id>/...` original upload
- `storage/datasets/<dataset_id>/train.jsonl`
- `storage/datasets/<dataset_id>/eval.jsonl`
- `storage/datasets/<dataset_id>/holdout.jsonl`
- `storage/models/<model_id>/sk_model.joblib` (when using `tfidf_logreg`)
- `storage/models/<model_id>/hf_model/` + `tokenizer/` (when using `hf_transformer`)
- `storage/models/<model_id>/metrics.json`
- `storage/models/<model_id>/false_negatives.json`
- `storage/runs/<run_id>/summary.json`
- `storage/runs/<run_id>/round_*.jsonl`
- `storage/runs/<run_id>/robustness_report.json`
- `storage/demo/demo_manifest.json` (written by demo runner)

## API usage quickstart

### Upload dataset

Multipart form:

- `file`: dataset file (`.csv` / `.json` / `.jsonl` / `SMSSpamCollection`)
- `options`: JSON string (see `DatasetUploadOptions`)

Example using curl:

```bash
curl -X POST "http://127.0.0.1:8000/dataset/upload" \
  -F "file=@SMSSpamCollection" \
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
