"""Retrieval metrics over ranked results, at the level of *sections*.

A section can be split into several chunks; counting each chunk separately
would reward a retriever for returning the same section three times. So each
expected section counts once: at the rank where it first appears.

- recall@k: fraction of the expected sections found in the top k.
- MRR (mean reciprocal rank): 1 / rank of the first relevant result. Rewards
  putting *something* right at the very top: what a user (or LLM) reads first.
- nDCG@k: rewards relevant results more the higher they rank, normalised so a
  perfect ordering scores 1.0. Sensitive to the whole ranking, not just the top.
"""

import math
from collections.abc import Sequence
from statistics import mean

from evals.dataset import Section


def first_ranks(
    ranked: Sequence[Section], expected: Sequence[Section], k: int
) -> dict[Section, int]:
    """1-based rank at which each expected section first appears in the top k."""
    wanted = set(expected)
    ranks: dict[Section, int] = {}
    for rank, section in enumerate(ranked[:k], start=1):
        if section in wanted and section not in ranks:
            ranks[section] = rank
    return ranks


def recall_at_k(ranked: Sequence[Section], expected: Sequence[Section], k: int) -> float:
    return len(first_ranks(ranked, expected, k)) / len(set(expected))


def reciprocal_rank(ranked: Sequence[Section], expected: Sequence[Section], k: int) -> float:
    ranks = first_ranks(ranked, expected, k)
    return 1 / min(ranks.values()) if ranks else 0.0


def ndcg_at_k(ranked: Sequence[Section], expected: Sequence[Section], k: int) -> float:
    gains = first_ranks(ranked, expected, k).values()
    dcg = sum(1 / math.log2(rank + 1) for rank in gains)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(set(expected)), k) + 1))
    return dcg / ideal


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return ordered[index]


def average(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def average_or_none(values: Sequence[float]) -> float | None:
    """For breakdowns: an empty group is "no data", not a score of zero."""
    return mean(values) if values else None
