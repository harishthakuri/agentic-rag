"""Summaries and Markdown tables from evaluation rows."""

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from evals.dataset import QuestionType
from evals.metrics import average, average_or_none, percentile
from evals.runner import AnswerRow, RetrievalRow


def retrieval_summary(rows: Sequence[RetrievalRow]) -> list[dict[str, Any]]:
    by_config: dict[str, list[RetrievalRow]] = defaultdict(list)
    for row in rows:
        by_config[row.config].append(row)
    summary = []
    for config, items in by_config.items():
        entry: dict[str, Any] = {
            "config": config,
            "recall@5": average([r.recall_at_5 for r in items]),
            "mrr@10": average([r.mrr_at_10 for r in items]),
            "ndcg@10": average([r.ndcg_at_10 for r in items]),
            "p50_ms": percentile([r.latency_ms for r in items], 50),
            "p95_ms": percentile([r.latency_ms for r in items], 95),
        }
        for qtype in (QuestionType.PARAPHRASE, QuestionType.EXACT, QuestionType.MULTI_PART):
            typed = [r.recall_at_5 for r in items if r.question_type == qtype.value]
            entry[f"recall@5:{qtype.value}"] = average_or_none(typed)
        summary.append(entry)
    return summary


def answer_summary(rows: Sequence[AnswerRow]) -> list[dict[str, Any]]:
    by_system: dict[str, list[AnswerRow]] = defaultdict(list)
    for row in rows:
        by_system[row.system].append(row)
    summary = []
    for system, items in by_system.items():
        answerable = [r for r in items if r.question_type != QuestionType.UNANSWERABLE.value]
        unanswerable = [r for r in items if r.question_type == QuestionType.UNANSWERABLE.value]
        judged = [r for r in items if r.faithfulness is not None]
        entry: dict[str, Any] = {
            "system": system,
            "correctness": average([r.correctness_score for r in answerable]),
            "fully_correct": _share(answerable, lambda r: r.correctness == "correct"),
            "faithfulness": average([r.faithfulness for r in judged if r.faithfulness is not None]),
            "answers_with_unsupported_claims": _share(judged, lambda r: bool(r.unsupported_claims)),
            "answers_with_citations": _share(
                [r for r in answerable if not r.declined], lambda r: r.cited_any
            ),
            "cited_expected_section": _share(answerable, lambda r: bool(r.cited_expected_section)),
            "declined_unanswerable": f"{sum(r.declined for r in unanswerable)}/{len(unanswerable)}",
            "wrongly_declined": sum(r.declined for r in answerable),
            "invalid_citations": sum(r.invalid_citations for r in items),
            "errors": sum(r.error is not None for r in items),
            "avg_tool_calls": average([float(r.tool_calls) for r in items]),
            "p50_s": percentile([r.latency_ms for r in items], 50) / 1000,
            "p95_s": percentile([r.latency_ms for r in items], 95) / 1000,
            "avg_tokens": average([float(r.prompt_tokens + r.completion_tokens) for r in items]),
        }
        for qtype in QuestionType:
            if qtype is QuestionType.UNANSWERABLE:
                continue
            typed = [r.correctness_score for r in answerable if r.question_type == qtype.value]
            entry[f"correctness:{qtype.value}"] = average_or_none(typed)
        summary.append(entry)
    return summary


def markdown(
    retrieval: list[dict[str, Any]], answers: list[dict[str, Any]], meta: dict[str, Any]
) -> str:
    lines = [f"# Evaluation: {meta['dataset']} ({meta['questions']} questions)", ""]
    lines.append(
        f"Collection `{meta['collection']}` · embeddings `{meta['embedding_model']}` · "
        f"chat `{meta['chat_model']}` · judge `{meta['judge_model']}` · {meta['finished_at']}"
    )
    if retrieval:
        lines += ["", "## Retrieval (answerable questions)", ""]
        lines += _table(
            [
                "Config",
                "Recall@5",
                "MRR@10",
                "nDCG@10",
                "Recall@5 paraphrase",
                "Recall@5 exact",
                "Recall@5 multi-part",
                "p50 ms",
                "p95 ms",
            ],
            [
                [
                    r["config"],
                    _f(r["recall@5"]),
                    _f(r["mrr@10"]),
                    _f(r["ndcg@10"]),
                    _f(r["recall@5:paraphrase"]),
                    _f(r["recall@5:exact"]),
                    _f(r["recall@5:multi_part"]),
                    f"{r['p50_ms']:.0f}",
                    f"{r['p95_ms']:.0f}",
                ]
                for r in retrieval
            ],
        )
    if answers:
        lines += ["", "## Answers", ""]
        lines += _table(
            [
                "System",
                "Correctness",
                "Fully correct",
                "Faithfulness",
                "Answers w/ unsupported claims",
                "Answers with citations",
                "Cited expected section",
                "Declined unanswerable",
                "Wrongly declined",
                "Invalid citations",
                "Avg tool calls",
                "p50 s",
                "p95 s",
                "Avg tokens",
            ],
            [
                [
                    a["system"],
                    _f(a["correctness"]),
                    _pct(a["fully_correct"]),
                    _f(a["faithfulness"]),
                    _pct(a["answers_with_unsupported_claims"]),
                    _pct(a["answers_with_citations"]),
                    _pct(a["cited_expected_section"]),
                    a["declined_unanswerable"],
                    str(a["wrongly_declined"]),
                    str(a["invalid_citations"]),
                    f"{a['avg_tool_calls']:.1f}",
                    f"{a['p50_s']:.1f}",
                    f"{a['p95_s']:.1f}",
                    f"{a['avg_tokens']:.0f}",
                ]
                for a in answers
            ],
        )
        lines += ["", "Correctness by question type:", ""]
        lines += _table(
            ["System", "Paraphrase", "Exact", "Multi-part"],
            [
                [
                    a["system"],
                    _f(a["correctness:paraphrase"]),
                    _f(a["correctness:exact"]),
                    _f(a["correctness:multi_part"]),
                ]
                for a in answers
            ],
        )
    return "\n".join(lines) + "\n"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _share(items: Sequence[AnswerRow], predicate: Callable[[AnswerRow], bool]) -> float:
    return sum(predicate(r) for r in items) / len(items) if items else 0.0


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _pct(value: float) -> str:
    return f"{value:.0%}"
