"""Opt-in checks against the real local models: `uv run pytest -m live`."""

import math

import pytest

from app.bootstrap.container import Container
from app.core.config import get_settings

pytestmark = pytest.mark.live


async def test_ollama_embeddings_have_configured_size_and_unit_length() -> None:
    container = Container(get_settings())
    try:
        [document] = await container.embedder.embed_documents(["A Service gives pods a stable IP."])
        query = await container.embedder.embed_query("How do pods get a stable address?")
        unrelated = await container.embedder.embed_query("How do I bake sourdough bread?")
    finally:
        await container.aclose()

    assert len(document) == container.settings.embedding_dim
    assert math.isclose(math.hypot(*document), 1.0, rel_tol=1e-6)

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    # The related question must be closer to the passage than the unrelated one.
    assert cosine(query, document) > cosine(unrelated, document)
