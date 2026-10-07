"""Domain errors. The presentation layer maps these to HTTP responses."""


class DomainError(Exception):
    """Base class for all business-rule violations."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DomainValidationError(DomainError):
    """A value violates a business invariant (e.g. an invalid collection name)."""


class NotFoundError(DomainError):
    """A requested entity does not exist."""


class ConflictError(DomainError):
    """The operation conflicts with existing state (e.g. a duplicate name)."""


class InvalidStateTransitionError(ConflictError):
    """An entity cannot move from its current state to the requested one."""


class AuthenticationError(DomainError):
    """Credentials are missing, unknown or revoked."""


__all__ = [
    "AuthenticationError",
    "ConflictError",
    "DomainError",
    "DomainValidationError",
    "InvalidStateTransitionError",
    "NotFoundError",
]
