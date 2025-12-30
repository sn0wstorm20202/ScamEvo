from __future__ import annotations

DEMO_DATASET_OPTIONS: dict[str, object] = {
    "train_ratio": 0.8,
    "eval_ratio": 0.1,
    "holdout_ratio": 0.1,
    "seed": 1337,
    "balance_strategy": "none",
}

DEMO_DETECTOR_TRAIN: dict[str, object] = {
    "backend": "tfidf_logreg",
    "seed": 1337,
    "detection_threshold": 0.5,
}

DEMO_DETECTOR_EVALUATE: dict[str, object] = {
    "split": "holdout",
    "detection_threshold": 0.5,
}

DEMO_ADVERSARIAL_RUN: dict[str, object] = {
    "rounds": 3,
    "seeds_per_round": 10,
    "candidates_per_seed": 3,
    "detection_threshold": 0.5,
    "similarity_threshold": 0.0,
    "require_anchors": False,
    "seed": 1337,
    "dry_run": False,
}

DEMO_ADVERSARIAL_RETRAIN: dict[str, object] = {
    "rounds": 1,
    "seeds_per_round": 10,
    "candidates_per_seed": 3,
    "similarity_threshold": 0.0,
    "require_anchors": False,
    "hard_max_examples": 25,
    "seed": 1337,
    "dry_run": False,
    "retrain_backend": "tfidf_logreg",
}
