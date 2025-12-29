from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source_filename TEXT,
                num_samples INTEGER,
                num_scam INTEGER,
                num_legit INTEGER
            );
            """
        )
        conn.execute(
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
        conn.execute(
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


def iter_rows(conn: sqlite3.Connection, query: str, params: tuple = ()) -> Iterator[sqlite3.Row]:
    cur = conn.execute(query, params)
    try:
        for row in cur:
            yield row
    finally:
        cur.close()
