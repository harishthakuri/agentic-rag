from typing import Protocol


class FileStorage(Protocol):
    """Raw uploaded files. Keys are relative, '/'-separated paths."""

    async def save(self, key: str, content: bytes) -> None: ...

    async def read(self, key: str) -> bytes: ...

    async def delete_prefix(self, prefix: str) -> None:
        """Delete every file under `prefix` (no error if nothing exists)."""
        ...
