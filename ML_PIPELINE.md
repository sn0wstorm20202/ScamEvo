# SCAM-EVO Backend — ML Pipeline (Detailed)

This document describes the ML flow implemented in the current backend MVP.

The current MVP uses **supervised** binary classification (`0` = legit, `1` = scam). **No unsupervised training** is used.

## 1) Dataset ingestion & preparation

### Inputs supported (current)

- `SMSSpamCollection` format (`ham<TAB>text` / `spam<TAB>text`)
- CSV datasets with detectable `text` + `label` columns
- JSON / JSONL datasets with `text` + `label` keys, or common variants

### Normalization rules (current)

- Convert labels to:
  - `0` = legit/ham
  - `1` = scam/spam/smishing/phishing
- Add canonical fields:
  - `id` (uuid)
  - `source` = `public_dataset`
  - `channel` = `sms`
  - `metadata` (extra columns for CSV or extra fields for JSON)

### Splits (current)

- `train.jsonl`
- `eval.jsonl`
- `holdout.jsonl`

Ratios and seed are controlled by `DatasetUploadOptions`.

### Outputs (current)

- `storage/datasets/<dataset_id>/meta.json`
- `storage/datasets/<dataset_id>/{train,eval,holdout}.jsonl`

## 2) Baseline detector (Defender)

### Model choice (current)

Two detector backends are supported:

- `tfidf_logreg` (recommended for demo/local)
  - TF-IDF vectorizer + Logistic Regression (scikit-learn)
  - Fast, offline, deterministic
- `hf_transformer` (optional)
  - HuggingFace `AutoModelForSequenceClassification`
  - Default base model: `distilbert-base-uncased`

### Training objective (current)

- Supervised binary classification.
- `tfidf_logreg`: Logistic Regression on TF-IDF features.
- `hf_transformer`: fine-tuning with `CrossEntropyLoss` with optional class weighting.

### Reproducibility (current)

- A fixed seed controls:
  - Python random
  - NumPy
  - Torch
  - cuDNN determinism flags
- Training config is stored per model:
  - `storage/models/<model_id>/train_config.json`

### Training artifacts (current)

- Common:
  - `storage/models/<model_id>/meta.json`
  - `storage/models/<model_id>/train_config.json`
  - `storage/models/<model_id>/metrics.json`
  - `storage/models/<model_id>/false_negatives.json` (written on evaluation)
- `tfidf_logreg`:
  - `storage/models/<model_id>/sk_model.joblib`
- `hf_transformer`:
  - `storage/models/<model_id>/hf_model/`
  - `storage/models/<model_id>/tokenizer/`

### Evaluation (current)

- Metrics:
  - accuracy
  - precision
  - recall
  - f1
- False negatives export:
  - saved to `storage/models/<model_id>/false_negatives.json` (after evaluation)

## 3) Explainability (initial)

### Token importance (current)

- In `POST /detector/infer` with `explain=true`:
  - `hf_transformer`: returns a simple mean-attention score per token (lightweight MVP signal)
  - `tfidf_logreg`: returns top contributing TF-IDF features (approx. feature contribution)

## 4) Generator (Attacker) — current (research-gated)

The generator is not a free-form LLM. It applies controlled transformations with strict constraints.

### Inputs

- `original_scam` text
- detector score
- mutation budget
- semantic similarity threshold

### Mutation actions (current)

- Lexical swaps (scam-preserving)
- Obfuscation of numbers/URLs
- Urgency modulation

### Validity filters (current)

- Similarity threshold: currently uses a lightweight TF-IDF cosine similarity filter.
- Optional scam anchors: CTA + urgency/authority/reward heuristics.
- Watermarks synthetic samples in generator metadata.

### Safety/ethics (current)

- Generator is gated behind `SCAMEVO_RESEARCH_MODE=1` and `SCAMEVO_DO_NOT_DEPLOY=1`.

## 5) Adversarial engine (core) — current (simulation)

Round-based mutation + scoring loop (simulation):

1. Sample scam seeds from a dataset split
2. Generate K mutations per seed
3. Score mutations with the detector
4. Select evasive samples (score below threshold)
5. Log everything (reproducible)

Note: full adversarial **retraining per round** is not implemented in this MVP. The current engine measures evasion rate and stores run artifacts.

Artifacts per run:

- `storage/runs/<run_id>/run_config.json`
- `storage/runs/<run_id>/round_0.jsonl`, `round_1.jsonl`, ...
- `storage/runs/<run_id>/summary.json`

## 6) Robustness evaluation — current (minimal)

The MVP robustness report is a compact summary derived from the stored run summary:

- total candidates
- evasive candidates
- evasion rate

## 7) Recommended operating mode (demo)

1. Upload dataset
2. Train baseline detector (`tfidf_logreg` is recommended for demo)
3. Evaluate on holdout
4. Run adversarial simulation
5. Generate robustness report
