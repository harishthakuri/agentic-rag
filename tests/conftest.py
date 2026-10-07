import pytest
from pydantic import SecretStr

from app.core.config import Environment, Settings


@pytest.fixture
def settings() -> Settings:
    """Settings isolated from the developer's `.env` file."""
    return Settings(
        _env_file=None,
        app_env=Environment.TEST,
        database_url=SecretStr("postgresql+asyncpg://user:pass@localhost:5432/test"),
    )
