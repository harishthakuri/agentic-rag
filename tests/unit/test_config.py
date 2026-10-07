import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings


def test_defaults_point_to_local_ollama(settings: Settings) -> None:
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_model == "gpt-oss:20b"
    assert settings.embedding_model == "qwen3-embedding:8b"
    assert settings.embedding_dim == 1024


def test_rejects_non_asyncpg_database_url() -> None:
    with pytest.raises(ValidationError, match="postgresql\\+asyncpg"):
        Settings(_env_file=None, database_url=SecretStr("postgresql://u:p@h/db"))


def test_rejects_embedding_dim_above_hnsw_limit() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            database_url=SecretStr("postgresql+asyncpg://u:p@h/db"),
            embedding_dim=4096,
        )


def test_database_url_is_not_leaked_in_repr(settings: Settings) -> None:
    assert "pass" not in repr(settings)
