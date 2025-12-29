from __future__ import annotations

from app.core.config import Settings
from app.db import postgres, sqlite


def init_db(settings: Settings) -> None:
    if settings.database_url:
        postgres.init_db(settings.database_url)
    else:
        sqlite.init_db(settings.db_path)


def insert_dataset(
    *,
    settings: Settings,
    dataset_id: str,
    version: str,
    created_at: str,
    source_filename: str,
    num_samples: int,
    num_scam: int,
    num_legit: int,
) -> None:
    if settings.database_url:
        with postgres.connect(settings.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO datasets (id, version, created_at, source_filename, num_samples, num_scam, num_legit) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (dataset_id, version, created_at, source_filename, num_samples, num_scam, num_legit),
                )
            conn.commit()
    else:
        with sqlite.connect(settings.db_path) as conn:
            conn.execute(
                "INSERT INTO datasets (id, version, created_at, source_filename, num_samples, num_scam, num_legit) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (dataset_id, version, created_at, source_filename, num_samples, num_scam, num_legit),
            )
            conn.commit()


def insert_model(
    *,
    settings: Settings,
    model_id: str,
    version: str,
    created_at: str,
    dataset_id: str,
    model_type: str,
    artifact_path: str,
    metrics_path: str,
) -> None:
    if settings.database_url:
        with postgres.connect(settings.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO models (id, version, created_at, dataset_id, model_type, artifact_path, metrics_path) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (model_id, version, created_at, dataset_id, model_type, artifact_path, metrics_path),
                )
            conn.commit()
    else:
        with sqlite.connect(settings.db_path) as conn:
            conn.execute(
                "INSERT INTO models (id, version, created_at, dataset_id, model_type, artifact_path, metrics_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (model_id, version, created_at, dataset_id, model_type, artifact_path, metrics_path),
            )
            conn.commit()


def insert_run(
    *,
    settings: Settings,
    run_id: str,
    created_at: str,
    run_type: str,
    config_path: str | None,
    artifacts_dir: str | None,
) -> None:
    if settings.database_url:
        with postgres.connect(settings.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO runs (id, created_at, run_type, config_path, artifacts_dir) VALUES (%s, %s, %s, %s, %s)",
                    (run_id, created_at, run_type, config_path, artifacts_dir),
                )
            conn.commit()
    else:
        with sqlite.connect(settings.db_path) as conn:
            conn.execute(
                "INSERT INTO runs (id, created_at, run_type, config_path, artifacts_dir) VALUES (?, ?, ?, ?, ?)",
                (run_id, created_at, run_type, config_path, artifacts_dir),
            )
            conn.commit()


def list_runs(*, settings: Settings, limit: int = 50) -> list[dict]:
    limit = int(limit)
    if limit <= 0:
        return []

    if settings.database_url:
        with postgres.connect(settings.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, created_at, run_type, config_path, artifacts_dir FROM runs ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "created_at": r[1],
                "run_type": r[2],
                "config_path": r[3],
                "artifacts_dir": r[4],
            }
            for r in rows
        ]

    with sqlite.connect(settings.db_path) as conn:
        rows = list(
            sqlite.iter_rows(
                conn,
                "SELECT id, created_at, run_type, config_path, artifacts_dir FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        )
    return [
        {
            "id": r["id"],
            "created_at": r["created_at"],
            "run_type": r["run_type"],
            "config_path": r["config_path"],
            "artifacts_dir": r["artifacts_dir"],
        }
        for r in rows
    ]


def get_run(*, settings: Settings, run_id: str) -> dict | None:
    if settings.database_url:
        with postgres.connect(settings.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, created_at, run_type, config_path, artifacts_dir FROM runs WHERE id = %s",
                    (run_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "run_type": row[2],
            "config_path": row[3],
            "artifacts_dir": row[4],
        }

    with sqlite.connect(settings.db_path) as conn:
        rows = list(
            sqlite.iter_rows(
                conn,
                "SELECT id, created_at, run_type, config_path, artifacts_dir FROM runs WHERE id = ?",
                (run_id,),
            )
        )
    if not rows:
        return None
    r = rows[0]
    return {
        "id": r["id"],
        "created_at": r["created_at"],
        "run_type": r["run_type"],
        "config_path": r["config_path"],
        "artifacts_dir": r["artifacts_dir"],
    }
