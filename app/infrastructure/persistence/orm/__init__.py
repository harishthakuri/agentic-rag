"""ORM entities. Import every entity here so Alembic autogenerate can see it."""

from app.infrastructure.persistence.orm.base import Base

__all__ = ["Base"]
