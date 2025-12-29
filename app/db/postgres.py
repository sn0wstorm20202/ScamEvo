from __future__ import annotations

from contextlib import contextmanager

try:
    import psycopg
except Exception:
    psycopg = None


@contextmanager
def connect(database_url: str):
    if psycopg is None:
        raise RuntimeError("psycopg is required for Postgres support. Install requirements and retry.")
    conn = psycopg.connect(database_url)
    try:
        yield conn
    finally:
        conn.close()


def init_db(database_url: str) -> None:
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    source_filename TEXT,
                    num_samples BIGINT,
                    num_scam BIGINT,
                    num_legit BIGINT
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    dataset_id TEXT,
                    model_type TEXT,
                    artifact_path TEXT,
                    metrics_path TEXT
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    run_type TEXT NOT NULL,
                    config_path TEXT,
                    artifacts_dir TEXT
                );
                """
            )
        conn.commit()
