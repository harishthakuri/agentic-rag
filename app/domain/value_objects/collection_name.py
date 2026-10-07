import re
from dataclasses import dataclass

from app.domain.exceptions import DomainValidationError

_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


@dataclass(frozen=True, slots=True)
class CollectionName:
    """URL-safe collection identifier, e.g. `kubernetes-docs`."""

    value: str

    def __post_init__(self) -> None:
        if not _PATTERN.fullmatch(self.value):
            raise DomainValidationError(
                "Collection name must be 3-64 characters: lowercase letters, digits, '-' or '_', "
                "starting with a letter or digit"
            )

    @classmethod
    def parse(cls, raw: str) -> "CollectionName":
        return cls(raw.strip().lower())

    def __str__(self) -> str:
        return self.value
