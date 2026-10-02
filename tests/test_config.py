"""Unit tests for configuration management."""

from pathlib import Path

from medical_coding.config.settings import Settings, get_settings


def test_default_settings() -> None:
    """Verify sensible offline defaults."""
    settings = Settings()
    assert settings.offline_mode is True
    assert settings.environment == "development"
    assert settings.llm_n_threads >= 1
    assert settings.llm_temperature == 0.0
    assert isinstance(settings.llm_model_path, Path)
    assert settings.max_batch_concurrency >= 10


def test_environment_override(monkeypatch) -> None:
    """Verify settings pick up env variables."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LLM_N_THREADS", "8")
    monkeypatch.setenv("LOG_LEVEL", "warning")
    monkeypatch.setenv("MAX_BATCH_CONCURRENCY", "20")

    settings = Settings()
    assert settings.environment == "production"
    assert settings.llm_n_threads == 8
    assert settings.log_level == "WARNING"
    assert settings.max_batch_concurrency == 20


def test_get_settings_singleton() -> None:
    """Verify cached singleton behavior."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()
