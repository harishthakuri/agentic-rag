"""Integration fixtures: a real PostgreSQL + pgvector in Docker.

The container is provisioned exactly like the real server: the same role
script (scripts/db/001_roles.sql) runs as superuser, so the tests exercise
the real least-privilege setup rather than a superuser shortcut.
"""

import socket
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from testcontainers.community.postgres import PostgresContainer

PGVECTOR_IMAGE = "pgvector/pgvector:pg16"
DB_NAME = "simple-rag-db"
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "db"

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
