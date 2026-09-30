"""Tests for Settings (pydantic-settings, RAG_ prefix)."""

from pathlib import Path

import pytest

from rag_generator.config import Settings

ENV_VARS = [
    "ANTHROPIC_API_KEY", "RAG_LLM_MODEL", "RAG_LLM_TEMPERATURE", "RAG_EMBED_MODEL",
    "RAG_DATA_DIR", "RAG_CHUNK_SIZE", "RAG_CHUNK_OVERLAP", "RAG_TOP_K", "RAG_MIN_SCORE",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def load_settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_settings_defaults_match_readme() -> None:
    settings = load_settings()
    assert settings.llm_model == "claude-haiku-4-5-20251001"
    assert settings.embed_model == "BAAI/bge-small-en-v1.5"
    assert settings.data_dir == Path("./data")
    assert (settings.chunk_size, settings.chunk_overlap) == (800, 100)
    assert (settings.top_k, settings.min_score) == (5, 0.3)


def test_settings_temperature_defaults_to_zero() -> None:
    assert load_settings().llm_temperature == 0


def test_settings_temperature_is_none_when_env_var_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_LLM_TEMPERATURE", "")
    assert load_settings().llm_temperature is None


def test_settings_temperature_reads_float_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_TEMPERATURE", "0.5")
    assert load_settings().llm_temperature == 0.5


def test_settings_reads_rag_prefixed_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_DATA_DIR", "/tmp/rag-data")
    settings = load_settings()
    assert (settings.top_k, settings.data_dir) == (7, Path("/tmp/rag-data"))


def test_settings_reads_unprefixed_anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    key = load_settings().anthropic_api_key
    assert key is not None and key.get_secret_value() == "sk-test"


def test_settings_api_key_is_hidden_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert "sk-test" not in repr(load_settings())


def test_settings_reads_values_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("RAG_MIN_SCORE=0.5\nANTHROPIC_API_KEY=sk-file\n")
    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]
    assert settings.min_score == 0.5
