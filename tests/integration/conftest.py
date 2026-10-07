"""Integration fixtures: a real PostgreSQL + pgvector in Docker.

The container is provisioned exactly like the real server: the same role
script (scripts/db/001_roles.sql) runs as superuser, so the tests exercise
the real least-privilege setup rather than a superuser shortcut.
"""

import socket
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import asyncpg
import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from testcontainers.community.postgres import PostgresContainer

from app.bootstrap.container import Container
from app.core.config import Environment, Settings
from app.main import create_app
from tests.fakes import FakeEmbedder

PGVECTOR_IMAGE = "pgvector/pgvector:pg16"
DB_NAME = "simple-rag-db"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "db"

OWNER_PASSWORD = "owner-test-pw"
APP_PASSWORD = "app-test-pw"


@dataclass(frozen=True)
class PostgresUrls:
    host: str
    port: int

    def dsn(self, user: str, password: str) -> str:
        return f"postgresql://{user}:{password}@{self.host}:{self.port}/{DB_NAME}"

    @property
    def owner(self) -> str:
        return self.dsn("simple_rag_owner", OWNER_PASSWORD)

    @property
    def app(self) -> str:
        return self.dsn("simple_rag_app", APP_PASSWORD)

    # SQLAlchemy URLs (asyncpg driver), as used in settings.
    @property
    def owner_sqlalchemy(self) -> str:
        return self.owner.replace("postgresql://", "postgresql+asyncpg://", 1)

    @property
    def app_sqlalchemy(self) -> str:
        return self.app.replace("postgresql://", "postgresql+asyncpg://", 1)


def _psql(container: PostgresContainer, *args: str) -> str:
    exit_code, output = container.exec(["psql", "-U", "postgres", "-d", DB_NAME, *args])
    text: str = output.decode()
    if exit_code != 0:
        raise RuntimeError(f"psql failed ({exit_code}):\n{text}")
    return text


def _wait_until_reachable(container: PostgresContainer, timeout_seconds: float = 30) -> None:
    """Wait for the final server (the init-phase one listens on a Unix socket only)
    and for the host port mapping (Docker Desktop / Rancher forward it asynchronously)."""
    host = container.get_container_host_ip()
    port = int(container.get_exposed_port(5432))
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        exit_code, _ = container.exec(["pg_isready", "-h", "127.0.0.1", "-U", "postgres"])
        if exit_code == 0:
            try:
                socket.create_connection((host, port), timeout=1).close()
                return
            except OSError:
                pass
        time.sleep(0.5)
    raise TimeoutError("PostgreSQL container did not become reachable in time")


@pytest.fixture(scope="session")
def postgres() -> Iterator[PostgresUrls]:
    container = PostgresContainer(
        PGVECTOR_IMAGE, username="postgres", password="postgres", dbname=DB_NAME
    ).with_volume_mapping(str(SCRIPTS_DIR), "/scripts", mode="ro")

    with container:
        _wait_until_reachable(container)
        _psql(container, "-c", "CREATE EXTENSION IF NOT EXISTS vector")
        _psql(container, "-v", "skip_passwords=1", "-f", "/scripts/001_roles.sql")
        _psql(
            container,
            "-c",
            f"ALTER ROLE simple_rag_owner PASSWORD '{OWNER_PASSWORD}';"
            f"ALTER ROLE simple_rag_app PASSWORD '{APP_PASSWORD}';",
        )
        yield PostgresUrls(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(5432)),
        )


@pytest.fixture(scope="session")
def migrated_postgres(postgres: PostgresUrls) -> PostgresUrls:
    """The container with every Alembic migration applied, as the owner role."""
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["database_url"] = postgres.owner_sqlalchemy
    config.attributes["configure_logging"] = False
    command.upgrade(config, "head")
    return postgres


# --- API over the migrated database -------------------------------------------
@dataclass
class Api:
    client: AsyncClient
    container: Container
    headers: dict[str, str]
    embedder: FakeEmbedder


@pytest.fixture
async def api(migrated_postgres: PostgresUrls, tmp_path: Path) -> AsyncIterator[Api]:
    """The real app (least-privilege DB role, real storage on a temp dir), with a
    deterministic fake embedder so tests don't need Ollama."""
    settings = Settings(
        _env_file=None,
        app_env=Environment.TEST,
        database_url=SecretStr(migrated_postgres.app_sqlalchemy),
        storage_dir=tmp_path / "uploads",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        container: Container = app.state.container
        embedder = FakeEmbedder(container.embedder.spec)
        container.embedder = embedder  # type: ignore[assignment]
        issued = await container.issue_api_key().execute("integration-tests")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield Api(client, container, {"Authorization": f"Bearer {issued.raw_key}"}, embedder)

    # Clean slate for the next test (deletes cascade to documents, chunks and jobs).
    conn = await asyncpg.connect(migrated_postgres.app)
    try:
        await conn.execute("DELETE FROM collections; DELETE FROM api_keys;")
    finally:
        await conn.close()
