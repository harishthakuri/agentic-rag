"""Domain errors. The presentation layer maps these to HTTP responses."""


class DomainError(Exception):
    """Base class for all business-rule violations."""


class NotFoundError(DomainError):
    """A requested entity does not exist."""


class ConflictError(DomainError):
    """The operation conflicts with existing state (e.g. a duplicate name)."""


__all__ = ["ConflictError", "DomainError", "NotFoundError"]
