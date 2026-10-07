import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID

from app.domain.exceptions import DomainValidationError
from app.domain.models.entity import Entity
from app.domain.value_objects import new_id

KEY_PREFIX = "arag_"
_DISPLAY_PREFIX_LENGTH = len(KEY_PREFIX) + 6
# Recording usage on every request would turn each read into a write.
_LAST_USED_RESOLUTION = timedelta(minutes=1)


def hash_api_key(raw_key: str) -> str:
    """SHA-256 is appropriate here, unlike for passwords: API keys carry 256 bits of
    randomness, so brute-forcing a hash is infeasible without slow key derivation."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


@dataclass(eq=False, kw_only=True, slots=True)
class ApiKey(Entity):
    id: UUID = field(default_factory=new_id)
    name: str
    display_prefix: str  # e.g. "arag_Ab12Cd": safe to show, identifies the key
    key_hash: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None

    @classmethod
    def issue(cls, name: str) -> tuple["ApiKey", str]:
        """Create a key. The raw secret is returned once and never stored."""
        if not name.strip():
            raise DomainValidationError("API key name must not be empty")
        raw_key = KEY_PREFIX + secrets.token_urlsafe(32)
        key = cls(
            name=name.strip(),
            display_prefix=raw_key[:_DISPLAY_PREFIX_LENGTH],
            key_hash=hash_api_key(raw_key),
        )
        return key, raw_key

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None

    def revoke(self) -> None:
        if self.revoked_at is None:
            self.revoked_at = datetime.now(UTC)

    def record_use(self, now: datetime | None = None) -> bool:
        """Update `last_used_at`; return True if it changed enough to be persisted."""
        now = now or datetime.now(UTC)
        if self.last_used_at and now - self.last_used_at < _LAST_USED_RESOLUTION:
            return False
        self.last_used_at = now
        return True
