import hashlib
import re
from dataclasses import dataclass

from app.domain.exceptions import DomainValidationError

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ContentHash:
    """SHA-256 of a document's raw bytes. Used to detect duplicate uploads."""

    value: str

    def __post_init__(self) -> None:
        if not _SHA256_HEX.fullmatch(self.value):
            raise DomainValidationError("Content hash must be a lowercase SHA-256 hex digest")

    @classmethod
    def of(cls, content: bytes) -> "ContentHash":
        return cls(hashlib.sha256(content).hexdigest())

    def __str__(self) -> str:
        return self.value
