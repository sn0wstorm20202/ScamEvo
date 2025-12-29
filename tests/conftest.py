from pathlib import Path
import pytest

@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    storage_dir = tmp_path / "storage"
    monkeypatch.setenv("SCAMEVO_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("SCAMEVO_DB_PATH", str(storage_dir / "metadata.sqlite3"))
    from app.core.config import get_settings

    get_settings.cache_clear()

    yield

    get_settings.cache_clear()
