"""Reciprocal Rank Fusion (RRF): merge several ranked lists into one.

Vector search scores (cosine similarity, 0..1) and keyword scores (ts_rank,
unbounded and corpus-dependent) are on different scales, so they can't simply
be added. RRF ignores the scores and uses only each item's *rank*:

    score(item) = Σ over lists  1 / (k + rank_in_that_list)

An item ranked 1st in one list scores 1/61 ≈ 0.0164 (with k = 60). An item
ranked 3rd in *both* lists scores 2/63 ≈ 0.0317 and wins: agreement between
independent retrievers is strong evidence of relevance. The constant k damps
the advantage of the very top ranks; 60 is the value from the original paper
(Cormack et al., 2009) and works well without tuning.
"""

from collections.abc import Hashable, Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion[K: Hashable](
    rankings: Sequence[Sequence[K]], *, k: int = DEFAULT_RRF_K
) -> list[tuple[K, float]]:
    """Fuse ranked lists (best first) into (item, score) pairs, best first.

    Ties keep the order in which items were first seen (earlier lists first),
    which makes the result deterministic.
    """
    if k < 0:
        raise ValueError("k must not be negative")
    scores: dict[K, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    first_seen = {item: index for index, item in enumerate(scores)}
    return sorted(scores.items(), key=lambda pair: (-pair[1], first_seen[pair[0]]))
