"""Cross-encoder reranker: a small model trained only to score (query, passage) pairs.

Like the LLM reranker, it reads the query and each passage *together*, but it
is a dedicated relevance model (e.g. BAAI/bge-reranker-v2-m3) rather than a
chat model asked to grade. It scores all candidates in one batched forward pass,
which takes milliseconds on a GPU instead of seconds, and its scores are fine
grained, so there are few ties. See docs/RERANKING.md.

Design choices:
- sentence-transformers (and PyTorch) is an optional extra: `uv sync --extra rerank`.
  It is imported only when the model is loaded.
- The model loads once (the API warms it up at startup) and stays in memory.
- `predict()` is synchronous GPU/CPU work, so it runs in a worker thread and
  never blocks the event loop. One prediction runs at a time: the model is
  shared, and batching already uses the hardware fully.
- Scores go through a sigmoid, so they are always 0..1 whatever the model's
  default. The order is unchanged; it only makes a fixed threshold meaningful
  (see ANSWER_MIN_CROSS_ENCODER_SCORE).
"""

import asyncio
from collections.abc import Callable, Sequence
from typing import Protocol

from app.application.ports.reranking import RerankCandidate, RerankError, RerankScore


class PairScorer(Protocol):
    """The slice of `sentence_transformers.CrossEncoder` this adapter uses."""

    def predict(
        self, inputs: list[tuple[str, str]], *, batch_size: int, show_progress_bar: bool
    ) -> Sequence[float]: ...


def load_cross_encoder(
    model: str, *, device: str | None = None, max_length: int | None = None
) -> PairScorer:
    """Load a sentence-transformers CrossEncoder (downloads from Hugging Face on first use)."""
    import torch  # optional dependency
    from sentence_transformers import CrossEncoder

    encoder: PairScorer = CrossEncoder(
        model, device=device, max_length=max_length, activation_fn=torch.nn.Sigmoid()
    )
    return encoder


class CrossEncoderReranker:
    def __init__(self, model: str, load: Callable[[], PairScorer], *, batch_size: int = 16) -> None:
        self._model_name = model
        self._load = load
        self._batch_size = batch_size
        self._scorer: PairScorer | None = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return f"cross_encoder:{self._model_name}"

    async def warm_up(self) -> None:
        """Load the model now instead of on the first search. Load errors propagate."""
        async with self._lock:
            await self._ensure_loaded()

    async def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> list[RerankScore]:
        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        async with self._lock:
            try:
                scorer = await self._ensure_loaded()
                scores = await asyncio.to_thread(
                    scorer.predict, pairs, batch_size=self._batch_size, show_progress_bar=False
                )
            except Exception as exc:
                raise RerankError(f"cross-encoder failed: {exc}") from exc
        if len(scores) != len(candidates):
            raise RerankError(f"cross-encoder returned {len(scores)} scores for {len(pairs)}")
        return [
            RerankScore(id=candidate.id, score=float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]

    async def _ensure_loaded(self) -> PairScorer:
        # Callers hold self._lock, so the model loads at most once.
        if self._scorer is None:
            self._scorer = await asyncio.to_thread(self._load)
        return self._scorer
