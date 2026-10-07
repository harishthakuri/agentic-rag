"""Declarative base for all ORM entities.

- Every table lives in the `rag` schema (owned by `simple_rag_owner`).
- A naming convention gives constraints/indexes deterministic names, so
  Alembic autogenerate produces stable, reviewable migrations.
"""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

DB_SCHEMA = "rag"

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(schema=DB_SCHEMA, naming_convention=NAMING_CONVENTION)
