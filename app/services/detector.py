from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
except Exception:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

from app.core.config import Settings
from app.db.metadata import insert_model
from app.schemas.detector import DetectorTrainRequest
from app.services.datasets import dataset_paths
from app.services.jsonl import read_jsonl


@dataclass(frozen=True)
class ModelPaths:
    root_dir: Path
    meta_path: Path
    hf_dir: Path
    tokenizer_dir: Path
    sklearn_model_path: Path
    metrics_path: Path
    false_negatives_path: Path
    train_config_path: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _require_hf_deps() -> None:
    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise RuntimeError(
            "hf_transformer backend requires 'torch' and 'transformers'. Install them or use backend='tfidf_logreg'."
        )


def model_paths(settings: Settings, model_id: str) -> ModelPaths:
    root = settings.models_dir / model_id
    return ModelPaths(
        root_dir=root,
        meta_path=root / "meta.json",
        hf_dir=root / "hf_model",
        tokenizer_dir=root / "tokenizer",
        sklearn_model_path=root / "sk_model.joblib",
        metrics_path=root / "metrics.json",
        false_negatives_path=root / "false_negatives.json",
        train_config_path=root / "train_config.json",
    )


def _set_seed(seed: int) -> None:
    _require_hf_deps()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _batch_tokenize(tokenizer, texts: list[str], max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def _predict_proba(model, tokenizer, texts: list[str], max_length: int, device: torch.device) -> tuple[list[float], list[list[tuple[str, float]]]]:
    model.eval()
    probs: list[float] = []
    explanations: list[list[tuple[str, float]]] = []

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc, output_attentions=True)
            logits = out.logits
            p = torch.softmax(logits, dim=-1)[0, 1].item()
            probs.append(float(p))

            token_scores: list[tuple[str, float]] = []
            if out.attentions is not None:
                att_stack = torch.stack(out.attentions, dim=0)
                att = att_stack.mean(dim=(0, 2))[0]
                scores = att.mean(dim=0).detach().cpu().numpy().tolist()
                tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].detach().cpu().tolist())
                for tok, score in zip(tokens, scores):
                    token_scores.append((tok, float(score)))

            explanations.append(token_scores)

    return probs, explanations


def _tfidf_explain(pipeline: Pipeline, text: str, top_k: int = 12) -> list[tuple[str, float]]:
    tfidf = pipeline.named_steps.get("tfidf")
    clf = pipeline.named_steps.get("clf")
    if not isinstance(tfidf, TfidfVectorizer) or not isinstance(clf, LogisticRegression):
        return []

    x = tfidf.transform([text])
    if x.shape[1] == 0:
        return []

    coef = clf.coef_[0]
    contrib = x.multiply(coef).toarray().reshape(-1)
    if contrib.size == 0:
        return []

    idx = np.argsort(-np.abs(contrib))[:top_k].tolist()
    feats = tfidf.get_feature_names_out()
    out: list[tuple[str, float]] = []
    for i in idx:
        if i < 0 or i >= len(feats):
            continue
        score = float(contrib[i])
        if score == 0.0:
            continue
        out.append((str(feats[i]), score))
    return out


def _train_tfidf_logreg(*, settings: Settings, req: DetectorTrainRequest) -> dict[str, Any]:
    dataset_meta_path = dataset_paths(settings, req.dataset_id).meta_path
    if not dataset_meta_path.exists():
        raise FileNotFoundError(f"Unknown dataset_id={req.dataset_id}")

    random.seed(req.seed)
    np.random.seed(req.seed)

    ds_paths = dataset_paths(settings, req.dataset_id)
    train_rows = list(read_jsonl(ds_paths.train_path))
    eval_rows = list(read_jsonl(ds_paths.eval_path))

    if not train_rows:
        raise ValueError("Train split is empty")

    texts_train = [r["text"] for r in train_rows]
    y_train = np.array([int(r["label"]) for r in train_rows], dtype=np.int64)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
            ("clf", LogisticRegression(max_iter=500, solver="liblinear", random_state=req.seed)),
        ]
    )
    pipeline.fit(texts_train, y_train)

    eval_use = eval_rows if eval_rows else train_rows
    texts_eval = [r["text"] for r in eval_use]
    y_true = np.array([int(r["label"]) for r in eval_use], dtype=np.int64)
    probs = pipeline.predict_proba(texts_eval)[:, 1].tolist()
    y_pred = (np.array(probs) >= float(req.detection_threshold)).astype(np.int64)

    metrics = {
        "backend": "tfidf_logreg",
        "num_samples": int(len(eval_use)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(eval_use) else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(eval_use) else 0.0,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(eval_use) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(eval_use) else 0.0,
        "threshold": float(req.detection_threshold),
    }

    model_id = str(uuid.uuid4())
    version = "v1"
    created_at = _utc_now_iso()

    paths = model_paths(settings, model_id)
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, paths.sklearn_model_path)

    train_cfg = req.model_dump()
    paths.train_config_path.write_text(json.dumps(train_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_obj = {"eval": metrics}
    paths.metrics_path.write_text(json.dumps(metrics_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "model_id": model_id,
        "version": version,
        "created_at": created_at,
        "dataset_id": req.dataset_id,
        "backend": "tfidf_logreg",
        "artifact_path": str(paths.root_dir),
    }
    paths.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    insert_model(
        settings=settings,
        model_id=model_id,
        version=version,
        created_at=created_at,
        dataset_id=req.dataset_id,
        model_type="tfidf_logreg",
        artifact_path=str(paths.root_dir),
        metrics_path=str(paths.metrics_path),
    )

    return {
        "model_id": model_id,
        "version": version,
        "created_at": created_at,
        "dataset_id": req.dataset_id,
        "base_model": req.base_model,
        "metrics": metrics_obj,
    }


def train_detector(*, settings: Settings, req: DetectorTrainRequest) -> dict[str, Any]:
    if getattr(req, "backend", "hf_transformer") == "tfidf_logreg":
        return _train_tfidf_logreg(settings=settings, req=req)

    _require_hf_deps()

    dataset_meta_path = dataset_paths(settings, req.dataset_id).meta_path
    if not dataset_meta_path.exists():
        raise FileNotFoundError(f"Unknown dataset_id={req.dataset_id}")

    _set_seed(req.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_paths = dataset_paths(settings, req.dataset_id)
    train_rows = list(read_jsonl(ds_paths.train_path))
    eval_rows = list(read_jsonl(ds_paths.eval_path))

    if not train_rows:
        raise ValueError("Train split is empty")

    tokenizer = AutoTokenizer.from_pretrained(req.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(req.base_model, num_labels=2)
    model.to(device)

    y_train = np.array([int(r["label"]) for r in train_rows], dtype=np.int64)
    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())
    if n0 == 0 or n1 == 0:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    else:
        w0 = (n0 + n1) / (2.0 * n0)
        w1 = (n0 + n1) / (2.0 * n1)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=req.learning_rate, weight_decay=req.weight_decay)

    def train_loader() -> Iterable[tuple[list[str], torch.Tensor]]:
        rows = train_rows.copy()
        random.Random(req.seed).shuffle(rows)
        for i in range(0, len(rows), req.batch_size):
            batch = rows[i : i + req.batch_size]
            texts = [r["text"] for r in batch]
            labels = torch.tensor([int(r["label"]) for r in batch], dtype=torch.long)
            yield texts, labels

    def eval_loader(rows: list[dict[str, Any]]) -> Iterable[tuple[list[str], torch.Tensor]]:
        for i in range(0, len(rows), req.batch_size):
            batch = rows[i : i + req.batch_size]
            texts = [r["text"] for r in batch]
            labels = torch.tensor([int(r["label"]) for r in batch], dtype=torch.long)
            yield texts, labels

    for _epoch in range(req.epochs):
        model.train()
        for texts, labels in train_loader():
            enc = _batch_tokenize(tokenizer, texts, req.max_length, device)
            labels = labels.to(device)
            out = model(**enc)
            loss = loss_fn(out.logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    train_metrics = evaluate_detector(
        settings=settings,
        model=model,
        tokenizer=tokenizer,
        rows=eval_rows if eval_rows else train_rows,
        max_length=req.max_length,
        detection_threshold=req.detection_threshold,
        device=device,
    )

    model_id = str(uuid.uuid4())
    version = "v1"
    created_at = _utc_now_iso()

    paths = model_paths(settings, model_id)
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(paths.hf_dir)
    tokenizer.save_pretrained(paths.tokenizer_dir)

    train_cfg = req.model_dump()
    paths.train_config_path.write_text(json.dumps(train_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "device": str(device),
        "eval": train_metrics,
    }
    paths.metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "model_id": model_id,
        "version": version,
        "created_at": created_at,
        "dataset_id": req.dataset_id,
        "base_model": req.base_model,
        "backend": "hf_transformer",
        "artifact_path": str(paths.root_dir),
    }
    paths.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    insert_model(
        settings=settings,
        model_id=model_id,
        version=version,
        created_at=created_at,
        dataset_id=req.dataset_id,
        model_type="hf_transformer",
        artifact_path=str(paths.root_dir),
        metrics_path=str(paths.metrics_path),
    )

    return {
        "model_id": model_id,
        "version": version,
        "created_at": created_at,
        "dataset_id": req.dataset_id,
        "base_model": req.base_model,
        "metrics": metrics,
    }


def load_model_meta(settings: Settings, model_id: str) -> dict[str, Any]:
    paths = model_paths(settings, model_id)
    if not paths.meta_path.exists():
        raise FileNotFoundError(f"Unknown model_id={model_id}")
    return json.loads(paths.meta_path.read_text(encoding="utf-8"))


def load_model_bundle(settings: Settings, model_id: str) -> dict[str, Any]:
    meta = load_model_meta(settings, model_id)
    backend = str(meta.get("backend") or "hf_transformer")
    paths = model_paths(settings, model_id)

    if backend == "tfidf_logreg":
        if not paths.sklearn_model_path.exists():
            raise FileNotFoundError(f"Unknown model_id={model_id}")
        pipeline = joblib.load(paths.sklearn_model_path)
        return {"backend": "tfidf_logreg", "pipeline": pipeline}

    _require_hf_deps()

    if not paths.hf_dir.exists() or not paths.tokenizer_dir.exists():
        raise FileNotFoundError(f"Unknown model_id={model_id}")
    tokenizer = AutoTokenizer.from_pretrained(paths.tokenizer_dir)
    model = AutoModelForSequenceClassification.from_pretrained(paths.hf_dir)
    return {"backend": "hf_transformer", "model": model, "tokenizer": tokenizer}


def evaluate_detector(
    *,
    settings: Settings,
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    max_length: int,
    detection_threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    if not rows:
        return {"num_samples": 0}

    texts = [r["text"] for r in rows]
    y_true = np.array([int(r["label"]) for r in rows], dtype=np.int64)

    probs, _ = _predict_proba(model, tokenizer, texts, max_length, device)
    y_pred = (np.array(probs) >= float(detection_threshold)).astype(np.int64)

    metrics = {
        "num_samples": int(len(rows)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(detection_threshold),
    }
    return metrics


def infer_detector(
    *,
    settings: Settings,
    model_id: str,
    texts: list[str],
    max_length: int | None = None,
    explain: bool = False,
) -> list[dict[str, Any]]:
    bundle = load_model_bundle(settings, model_id)

    if bundle["backend"] == "tfidf_logreg":
        pipeline = bundle["pipeline"]
        probs = pipeline.predict_proba(texts)[:, 1].tolist()
        explanations = [_tfidf_explain(pipeline, t) if explain else [] for t in texts]
    else:
        _require_hf_deps()
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if max_length is None:
            max_length = 192

        probs, explanations = _predict_proba(model, tokenizer, texts, max_length, device)

    items: list[dict[str, Any]] = []
    for text, p, token_scores in zip(texts, probs, explanations):
        item: dict[str, Any] = {
            "text": text,
            "scam_probability": float(p),
            "prediction": 1 if float(p) >= 0.5 else 0,
        }
        if explain:
            item["token_importance"] = [{"token": t, "score": s} for t, s in token_scores]
        items.append(item)

    return items


def evaluate_model_on_dataset(
    *,
    settings: Settings,
    model_id: str,
    dataset_id: str,
    split: str,
    detection_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundle = load_model_bundle(settings, model_id)

    ds = dataset_paths(settings, dataset_id)
    path = {"train": ds.train_path, "eval": ds.eval_path, "holdout": ds.holdout_path}[split]
    rows = list(read_jsonl(path))

    texts = [r["text"] for r in rows]
    if bundle["backend"] == "tfidf_logreg":
        pipeline = bundle["pipeline"]
        probs = pipeline.predict_proba(texts)[:, 1].tolist() if len(texts) else []
    else:
        _require_hf_deps()
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        probs, _ = _predict_proba(model, tokenizer, texts, 192, device)
    y_true = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    y_pred = (np.array(probs) >= float(detection_threshold)).astype(np.int64)

    metrics = {
        "num_samples": int(len(rows)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(rows) else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(rows) else 0.0,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(rows) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(rows) else 0.0,
        "threshold": float(detection_threshold),
    }

    false_negatives: list[dict[str, Any]] = []
    for r, p, pred in zip(rows, probs, y_pred.tolist()):
        if int(r["label"]) == 1 and int(pred) == 0:
            false_negatives.append({"id": r.get("id"), "text": r.get("text"), "scam_probability": float(p)})

    paths = model_paths(settings, model_id)
    if paths.false_negatives_path.parent.exists():
        paths.false_negatives_path.write_text(json.dumps(false_negatives, ensure_ascii=False, indent=2), encoding="utf-8")

    return metrics, false_negatives
