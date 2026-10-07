"""Verifies the least-privilege guarantees of scripts/db/001_roles.sql."""

from collections.abc import AsyncIterator

import asyncpg
import pytest

from tests.integration.conftest import PostgresUrls

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
async def owner_created_table(postgres: PostgresUrls) -> AsyncIterator[None]:
    """Simulates a migration: the owner creates a table after the script ran."""
    conn = await asyncpg.connect(postgres.owner)
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS probe (
                id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                embedding vector(3)
            )
            """
        )
        yield
        await conn.execute("DROP TABLE probe")  # the shared database must match the models
    finally:
        await conn.close()


async def test_roles_default_to_rag_schema(postgres: PostgresUrls) -> None:
    for dsn in (postgres.owner, postgres.app):
        conn = await asyncpg.connect(dsn)
        try:
            assert await conn.fetchval("SHOW search_path") == "rag, public"
        finally:
            await conn.close()


async def test_app_can_read_write_and_vector_search_new_tables(postgres: PostgresUrls) -> None:
    conn = await asyncpg.connect(postgres.app)
    try:
        await conn.execute("INSERT INTO probe (embedding) VALUES ('[1,0,0]'), ('[0,1,0]')")
        nearest = await conn.fetchval(
            "SELECT embedding::text FROM probe ORDER BY embedding <=> '[0.9,0.1,0]' LIMIT 1"
        )
        assert nearest == "[1,0,0]"
        await conn.execute("DELETE FROM probe")
    finally:
        await conn.close()


@pytest.mark.parametrize(
    "ddl",
    [
        "CREATE TABLE rag.sneaky (id int)",
        "CREATE TABLE public.sneaky (id int)",
        "ALTER TABLE probe ADD COLUMN sneaky int",
        "DROP TABLE probe",
        "TRUNCATE probe",
    ],
)
async def test_app_cannot_run_ddl(postgres: PostgresUrls, ddl: str) -> None:
    conn = await asyncpg.connect(postgres.app)
    try:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute(ddl)
    finally:
        await conn.close()


async def test_owner_cannot_create_objects_in_public(postgres: PostgresUrls) -> None:
    conn = await asyncpg.connect(postgres.owner)
    try:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute("CREATE TABLE public.sneaky (id int)")
    finally:
        await conn.close()


async def test_app_statement_timeout_is_enforced(postgres: PostgresUrls) -> None:
    conn = await asyncpg.connect(postgres.app)
    try:
        assert await conn.fetchval("SHOW statement_timeout") == "30s"
    finally:
        await conn.close()
