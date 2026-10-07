from dataclasses import dataclass

from app.domain.exceptions import DomainValidationError

MAX_PAGE_SIZE = 100


@dataclass(frozen=True, slots=True)
class PageRequest:
    limit: int = 50
    offset: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.limit <= MAX_PAGE_SIZE:
            raise DomainValidationError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        if self.offset < 0:
            raise DomainValidationError("offset must not be negative")
