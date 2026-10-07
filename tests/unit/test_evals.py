import json
import math
from pathlib import Path

import pytest

from evals.dataset import EvalQuestion, QuestionType, load_dataset, section_of
from evals.metrics import ndcg_at_k, percentile, recall_at_k, reciprocal_rank

A, B, C, X = ("Doc", "A"), ("Doc", "B"), ("Doc", "C"), ("Other", "X")


def test_recall_counts_each_expected_section_once() -> None:
    ranked = [A, A, X, B]  # A appears twice (two chunks of one section)
    assert recall_at_k(ranked, [A, B], 5) == 1.0
    assert recall_at_k(ranked, [A, B], 2) == 0.5
    assert recall_at_k(ranked, [A, B, C], 5) == pytest.approx(2 / 3)


def test_reciprocal_rank_uses_first_relevant_hit() -> None:
    assert reciprocal_rank([X, B, A], [A, B], 10) == 0.5
    assert reciprocal_rank([X, X], [A], 10) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking_and_lower_otherwise() -> None:
    assert ndcg_at_k([A, B, X], [A, B], 10) == 1.0
    # Relevant results at ranks 2 and 3 instead of 1 and 2:
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([X, A, B], [A, B], 10) == pytest.approx(expected)
    # A duplicate chunk of an already-counted section earns nothing.
    assert ndcg_at_k([A, A, B], [A, B], 10) == pytest.approx(
        (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    )


def test_percentile_nearest_rank() -> None:
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile([5, 1, 3, 2, 4], 95) == 5
    assert percentile([], 50) == 0.0


def test_section_of_joins_heading_path() -> None:
    assert section_of("HTTP Caching", ("Validation", "ETag")) == (
        "HTTP Caching",
        "Validation > ETag",
    )
    assert section_of("HTTP Caching", ()) == ("HTTP Caching", "")


def test_dataset_rules(tmp_path: Path) -> None:
    def line(qid: str, qtype: str, expected: list[list[str]]) -> str:
        return json.dumps(
            {"id": qid, "type": qtype, "question": "?", "expected": expected, "reference": "r"}
        )

    good = line("q1", "exact", [["D", "S"]])
    unanswerable_with_expected = line("q2", "unanswerable", [["D", "S"]])
    path = tmp_path / "set.jsonl"

    path.write_text(good + "\n" + good)
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)

    path.write_text(unanswerable_with_expected)
    with pytest.raises(ValueError, match="expected"):
        load_dataset(path)


def test_shipped_dataset_is_valid() -> None:
    questions = load_dataset(Path("evals/datasets/samples.jsonl"))
    assert len(questions) >= 20
    assert {q.type for q in questions} == set(QuestionType)
    assert all(isinstance(q, EvalQuestion) for q in questions)
