"""Uploaded files on the local filesystem.

Swappable for S3/MinIO behind the same FileStorage port.
"""

import asyncio
import shutil
from pathlib import Path


class LocalFileStorage:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    async def save(self, key: str, content: bytes) -> None:
        path = self._path(key)
        await asyncio.to_thread(_write_atomically, path, content)

    async def read(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)

    async def delete_prefix(self, prefix: str) -> None:
        target = self._path(prefix.rstrip("/"))
        if target == self._root:
            raise ValueError("Refusing to delete the storage root")
        await asyncio.to_thread(shutil.rmtree, target, ignore_errors=True)

    def _path(self, key: str) -> Path:
        """Resolve a key inside the root, rejecting traversal such as '../../etc/passwd'."""
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError(f"Storage key escapes the storage root: {key!r}")
        return path


def _write_atomically(path: Path, content: bytes) -> None:
    # Write to a temp file and rename: readers never see a half-written file.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(content)
    tmp.replace(path)
