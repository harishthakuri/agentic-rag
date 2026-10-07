"""Alembic environment (async).

Connects with MIGRATIONS_DATABASE_URL, the schema-owner role. The runtime
role (DATABASE_URL) has no DDL rights and is never used here.
"""

import asyncio
from logging.config import fileConfig
from typing import Any, Literal

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from app.core.config import get_settings
from app.infrastructure.persistence import orm  # noqa: F401  (registers every entity)
from app.infrastructure.persistence.orm.base import DB_SCHEMA, Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _migrations_url() -> str:
    settings = get_settings()
    if settings.migrations_database_url is None:
        raise RuntimeError("MIGRATIONS_DATABASE_URL is not set (see .env.example)")
    return settings.migrations_database_url.get_secret_value()


def _include_name(
    name: str | None,
    type_: Literal[
        "schema", "table", "column", "index", "unique_constraint", "foreign_key_constraint"
    ],
    parent_names: Any,
) -> bool:
    # Autogenerate only looks at our schema, never at `public` (pgvector lives there).
    if type_ == "schema":
        return name == DB_SCHEMA
    return True


def _configure(**kwargs: Any) -> None:
    context.configure(
        target_metadata=target_metadata,
        version_table_schema=DB_SCHEMA,
        include_schemas=True,
        include_name=_include_name,
        compare_type=True,
        compare_server_default=True,
        **kwargs,
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (`alembic upgrade head --sql`)."""
    _configure(url=_migrations_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def _run_sync_migrations(connection: Connection) -> None:
    _configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_migrations_url(), poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_run_sync_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
