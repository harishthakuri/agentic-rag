"""Run the evaluation: retrieval configurations and answering systems."""

import time
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import UUID

from app.application.use_cases.agent import AgentQuery
from app.application.use_cases.answering import AskQuery
from app.application.use_cases.retrieval import SearchMode, SearchQuery
from app.bootstrap.container import Container
from evals.dataset import EvalQuestion, Section, section_of
from evals.judge import Judge, Verdict
from evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

DECLINE_MARKERS = ("couldn't find", "could not find", "not in the documents")


# --- Retrieval ------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    mode: SearchMode
    rerank: bool


RETRIEVAL_CONFIGS = [
    RetrievalConfig("vector", SearchMode.VECTOR, rerank=False),
    RetrievalConfig("keyword", SearchMode.KEYWORD, rerank=False),
    RetrievalConfig("hybrid", SearchMode.HYBRID, rerank=False),
    RetrievalConfig("hybrid+rerank", SearchMode.HYBRID, rerank=True),
]


@dataclass
class RetrievalRow:
    question_id: str
    question_type: str
    config: str
    recall_at_5: float
    mrr_at_10: float
    ndcg_at_10: float
    latency_ms: float
    ranked: list[Section]


async def evaluate_retrieval(
    container: Container,
    collection_id: UUID,
    questions: list[EvalQuestion],
    configs: list[RetrievalConfig],
    progress: Any = print,
) -> list[RetrievalRow]:
    search = container.search_collection()
    rows: list[RetrievalRow] = []
    for config in configs:
        for q in (q for q in questions if q.answerable):
            result = await search.execute(
                SearchQuery(
                    collection_id, q.question, mode=config.mode, top_k=10, rerank=config.rerank
                )
            )
            ranked = [section_of(h.match.document_title, h.match.heading_path) for h in result.hits]
            rows.append(
                RetrievalRow(
                    question_id=q.id,
                    question_type=q.type.value,
                    config=config.name,
                    recall_at_5=recall_at_k(ranked, q.expected, 5),
                    mrr_at_10=reciprocal_rank(ranked, q.expected, 10),
                    ndcg_at_10=ndcg_at_k(ranked, q.expected, 10),
                    latency_ms=result.timings_ms["total"],
                    ranked=ranked,
                )
            )
        progress(f"  retrieval: {config.name} done")
    return rows


# --- Answers ----------------------------------------------------------------------
@dataclass
class AnswerRow:
    question_id: str
    question_type: str
    system: str
    answer: str
    declined: bool
    correctness: str
    correctness_score: float
    correctness_reason: str
    faithfulness: float | None  # None when the answer declined (no claims to check)
    claims: int
    unsupported_claims: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # "[n] Document > Section", to audit the judge
    cited_expected_section: bool | None = None  # None for unanswerable questions
    cited_any: bool = False  # the answer cites at least one valid source
    invalid_citations: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Answer:
    text: str
    sources: list[tuple[int, Section, str]]  # number, section, text
    cited: list[int]
    invalid: list[int]
    tool_calls: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


async def _ask(container: Container, collection_id: UUID, question: str) -> _Answer:
    result = await container.ask_question().execute(AskQuery(collection_id, question))
    return _Answer(
        text=result.answer,
        sources=[
            (s.number, section_of(s.document_title, s.heading_path), s.text) for s in result.sources
        ],
        cited=result.cited,
        invalid=result.invalid_citations,
        tool_calls=0,
        latency_ms=result.timings_ms["total"],
        prompt_tokens=result.usage.prompt_tokens if result.usage else 0,
        completion_tokens=result.usage.completion_tokens if result.usage else 0,
    )


async def _agent(container: Container, collection_id: UUID, question: str) -> _Answer:
    completed, _ = await container.agent_ask().execute(AgentQuery(collection_id, question))
    return _Answer(
        text=completed.answer,
        sources=[
            (s.number, section_of(s.match.document_title, s.match.heading_path), s.match.text)
            for s in completed.sources
        ],
        cited=completed.cited,
        invalid=completed.invalid_citations,
        tool_calls=completed.tool_calls,
        latency_ms=completed.timings_ms["total"],
        prompt_tokens=completed.usage.prompt_tokens,
        completion_tokens=completed.usage.completion_tokens,
    )


SYSTEMS = {"ask": _ask, "agent": _agent}


async def evaluate_answers(
    container: Container,
    collection_id: UUID,
    questions: list[EvalQuestion],
    systems: list[str],
    judge: Judge,
    progress: Any = print,
) -> list[AnswerRow]:
    rows: list[AnswerRow] = []
    for system in systems:
        for i, q in enumerate(questions, start=1):
            started = time.perf_counter()
            try:
                answer = await SYSTEMS[system](container, collection_id, q.question)
                rows.append(await _grade(judge, q, system, answer))
            except Exception as exc:  # one failure must not sink the whole run
                rows.append(_failed(q, system, exc))
            progress(
                f"  {system} {i}/{len(questions)} {q.id}: {rows[-1].correctness} "
                f"({time.perf_counter() - started:.0f}s)"
            )
    return rows


async def _grade(judge: Judge, q: EvalQuestion, system: str, answer: _Answer) -> AnswerRow:
    # Models use typographic apostrophes too: "couldn\u2019t" (right single quote).
    normalized = answer.text.lower().replace("\u2019", "'")
    declined = any(marker in normalized for marker in DECLINE_MARKERS)
    correctness = await judge.correctness(q.question, q.reference, answer.text)

    faithfulness = None
    claims, unsupported = 0, []
    if not declined:
        verdict = await judge.faithfulness(
            q.question, [(n, text) for n, _, text in answer.sources], answer.text
        )
        faithfulness, claims, unsupported = verdict.score, verdict.total, verdict.unsupported

    cited_sections = {section for n, section, _ in answer.sources if n in answer.cited}
    return AnswerRow(
        question_id=q.id,
        question_type=q.type.value,
        system=system,
        answer=answer.text,
        declined=declined,
        correctness=correctness.verdict.value,
        correctness_score=correctness.verdict.score,
        correctness_reason=correctness.reason,
        faithfulness=faithfulness,
        claims=claims,
        unsupported_claims=unsupported,
        sources=[
            f"[{n}] {doc} > {section}".rstrip(" >") for n, (doc, section), _ in answer.sources
        ],
        cited_expected_section=bool(cited_sections & set(q.expected)) if q.answerable else None,
        cited_any=bool(answer.cited),
        invalid_citations=len(answer.invalid),
        tool_calls=answer.tool_calls,
        latency_ms=answer.latency_ms,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
    )


def _failed(q: EvalQuestion, system: str, exc: Exception) -> AnswerRow:
    return AnswerRow(
        question_id=q.id,
        question_type=q.type.value,
        system=system,
        answer="",
        declined=False,
        correctness=Verdict.INCORRECT.value,
        correctness_score=0.0,
        correctness_reason="system error",
        faithfulness=None,
        claims=0,
        error=f"{type(exc).__name__}: {exc}"[:500],
    )
