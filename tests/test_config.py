# tests/test_config.py
from app.config import get_settings


def test_defaults_load_without_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = get_settings()
    assert settings.db_path == "data/facts.db"
    assert settings.cache_dir == "data/cache"
    assert settings.gemini_api_key is None
    assert settings.magnitude_cap == 0.5
    assert settings.vintage_gap_days == 90


def test_env_override(monkeypatch):
    monkeypatch.setenv("FKL_DB_PATH", "custom.db")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    from app.config import Settings
    settings = Settings()
    assert settings.db_path == "custom.db"
    assert settings.gemini_api_key == "test-key-123"
