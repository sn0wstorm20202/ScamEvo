from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_dotenv_if_present(base_dir: Path) -> None:
    dotenv_path = base_dir / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.startswith(('"', "'")) and value.endswith(('"', "'")) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


@dataclass(frozen=True)
class Settings:
    app_name: str
    base_dir: Path

    storage_dir: Path
    raw_datasets_dir: Path
    datasets_dir: Path
    models_dir: Path
    runs_dir: Path

    database_url: str | None
    db_path: Path

    research_mode: bool
    do_not_deploy: bool

    demo_mode: bool

    generator_backend: str
    generator_mode: str

    llm_provider: str
    llm_model: str
    openai_api_key: str | None
    openai_base_url: str

    llm_timeout_seconds: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[2]
    _load_dotenv_if_present(base_dir)
    storage_dir = Path(os.getenv("SCAMEVO_STORAGE_DIR", str(base_dir / "storage")))

    raw_datasets_dir = storage_dir / "raw_datasets"
    datasets_dir = storage_dir / "datasets"
    models_dir = storage_dir / "models"
    runs_dir = storage_dir / "runs"

    database_url = os.getenv("SCAMEVO_DATABASE_URL")
    if database_url is not None:
        database_url = database_url.strip() or None
    db_path = Path(os.getenv("SCAMEVO_DB_PATH", str(storage_dir / "metadata.sqlite3")))

    generator_backend = str(os.getenv("SCAMEVO_GENERATOR_BACKEND", "rule")).strip().lower()
    if generator_backend not in {"rule", "llm"}:
        generator_backend = "rule"

    generator_mode = str(os.getenv("SCAMEVO_GENERATOR_MODE", "paraphrase_with_obfuscation")).strip()
    if not generator_mode:
        generator_mode = "paraphrase_with_obfuscation"

    llm_provider = str(os.getenv("SCAMEVO_LLM_PROVIDER", "openai")).strip().lower()
    if not llm_provider:
        llm_provider = "openai"

    llm_model = str(os.getenv("SCAMEVO_LLM_MODEL", "gpt-5-mini")).strip()
    if not llm_model:
        llm_model = "gpt-5-mini"

    openai_api_key = os.getenv("SCAMEVO_OPENAI_API_KEY")
    if openai_api_key is not None:
        openai_api_key = openai_api_key.strip() or None

    openai_base_url = str(os.getenv("SCAMEVO_OPENAI_BASE_URL", "https://api.openai.com/v1")).strip().rstrip("/")
    if not openai_base_url:
        openai_base_url = "https://api.openai.com/v1"

    try:
        llm_timeout_seconds = float(os.getenv("SCAMEVO_LLM_TIMEOUT_SECONDS", "30"))
    except Exception:
        llm_timeout_seconds = 30.0
    if llm_timeout_seconds <= 0:
        llm_timeout_seconds = 30.0

    return Settings(
        app_name=os.getenv("SCAMEVO_APP_NAME", "SCAM-EVO Backend"),
        base_dir=base_dir,
        storage_dir=storage_dir,
        raw_datasets_dir=raw_datasets_dir,
        datasets_dir=datasets_dir,
        models_dir=models_dir,
        runs_dir=runs_dir,
        database_url=database_url,
        db_path=db_path,
        research_mode=_env_bool("SCAMEVO_RESEARCH_MODE", True),
        do_not_deploy=_env_bool("SCAMEVO_DO_NOT_DEPLOY", True),
        demo_mode=_env_bool("SCAMEVO_DEMO_MODE", False),

        generator_backend=generator_backend,
        generator_mode=generator_mode,
        llm_provider=llm_provider,
        llm_model=llm_model,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        llm_timeout_seconds=llm_timeout_seconds,
    )
