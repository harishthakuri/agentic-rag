from typing import Protocol


class TokenCounter(Protocol):
    """Models limit and bill by tokens (≈ 4 characters of English), not characters."""

    def count(self, text: str) -> int: ...

    def split(self, text: str, max_tokens: int) -> list[str]:
        """Hard-split text into pieces of at most `max_tokens` (last resort)."""
        ...
