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

For best offline performance on SMS-style spam/scam text, TF-IDF with **character n-grams** (e.g. `char_wb`) often outperforms word-only features.

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

## 2.1) Unified dataset training (global model)

The backend can train per uploaded dataset, and also supports building a **unified** dataset from multiple sources (e.g., multiple files under `datasets/`).

Script:

- `scripts/train_unified_model.py`

Outputs:

- `storage/unified/unified_<profile>_manifest.json`
- unified dataset saved under `storage/datasets/<dataset_id>/...`

## 2.2) Best-model search (offline)

For best results without external downloads, the repo includes an offline model search script:

- `scripts/train_best_model.py`

It trains multiple `tfidf_logreg` variants (word vs `char_wb`, different n-gram ranges, class weighting, regularization) and optionally tunes the decision threshold on the eval split.

Outputs:

- `storage/best/best_<profile>_manifest.json`

## 2.3) Best-model search (GridSearchCV)

For cross-validated hyperparameter tuning, the repo also includes a `GridSearchCV`-based trainer for `tfidf_logreg`.

- `scripts/train_best_model_cv.py`

It runs `GridSearchCV` on the training split (stratified K-fold), then (optionally) tunes the decision threshold on the eval split, and evaluates once on holdout.

Outputs:

- `storage/best/best_cv_<profile>_manifest.json`

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

### Generator backends (current)

- `SCAMEVO_GENERATOR_BACKEND=rule` (default)
  - Fully offline rule-based mutations.
- `SCAMEVO_GENERATOR_BACKEND=llm` (optional)
  - OpenAI-based mutation generation.
  - Requires `SCAMEVO_OPENAI_API_KEY`.

## 5) Adversarial engine (core) — current (simulation)

Round-based mutation + scoring loop (simulation):

1. Sample scam seeds from a dataset split
2. Generate K mutations per seed
3. Score mutations with the detector
4. Select evasive samples (score below threshold)
5. Log everything (reproducible)

Note: the simulation endpoint measures evasion rate and stores run artifacts. Adversarial retraining is implemented separately via `POST /adversarial/retrain`.

Artifacts per run:

- `storage/runs/<run_id>/run_config.json`
- `storage/runs/<run_id>/round_0.jsonl`, `round_1.jsonl`, ...
- `storage/runs/<run_id>/summary.json`

## 5.1) Adversarial retraining (Defender learns from attacks) — current

The backend supports a true adversarial retraining loop:

1. Generate candidate mutations (attacker)
2. Score candidates using the current detector + threshold
3. Select hard examples (evasive candidates)
4. Create a new augmented dataset version by appending synthetic hard examples to the train split
5. Retrain a new detector model on the augmented dataset
6. Re-evaluate baseline vs attacked vs defended and compute deltas

Endpoint:

- `POST /adversarial/retrain`

Artifacts per retrain run:

- `storage/runs/<run_id>/run_config.json`
- `storage/runs/<run_id>/round_*.jsonl`
- `storage/runs/<run_id>/hard_examples.jsonl`
- `storage/runs/<run_id>/summary.json`

Dataset augmentation output:

- New dataset id with a `parent_dataset_id` reference in its `meta.json`.
- Only the train split is augmented; eval/holdout are preserved.

## 6) Robustness evaluation — current (minimal)

The robustness report is derived from the stored run summary:

- total candidates
- evasive candidates
- evasion rate

For `adversarial_retrain` runs, the report also includes:

- `baseline_eval`: holdout metrics on the original model
- `attacked`: metrics over generated attack candidates
- `defended_eval`: holdout metrics on the defended model
- `defended_attack`: evasion rate of the defended model on the same attack set
- `delta`: defended vs baseline deltas (including evasion-rate reduction)

Endpoint:

- `GET /robustness/report?run_id=...`

## 7) Recommended operating mode (demo)

1. Upload dataset
2. Train baseline detector (`tfidf_logreg` is recommended for demo)
3. Evaluate on holdout
4. Run adversarial simulation (`POST /adversarial/run`)
5. Run adversarial retraining (`POST /adversarial/retrain`)
6. Generate robustness reports (`GET /robustness/report`)

The repo also includes an end-to-end demo runner:

- `scripts/run_demo_story.py`

It produces:

- `storage/demo/<retrain_run_id>/experiment_manifest.json`
- `storage/demo/<retrain_run_id>/robustness_report.json`
- `storage/demo/<retrain_run_id>/robustness_attack_report.json`
- `storage/demo/<retrain_run_id>/summary.md`

## 8) Threshold persistence (consistency)

Detection thresholds are persisted per model:

- `storage/models/<model_id>/train_config.json` contains `detection_threshold`.

Inference/evaluation default to the persisted threshold unless you override via request.
