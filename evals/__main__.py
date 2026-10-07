"""Evaluate retrieval and answer quality against a labelled question set.

    uv run python -m evals                                  # everything (slow: LLM calls)
    uv run python -m evals --skip-answers                   # retrieval only (~2 min)
    uv run python -m evals --systems ask --limit 5          # quick smoke run
    uv run python -m evals --judge-model gpt-5-mini         # judge with another model

Writes evals/results/<timestamp>.json (every row) and .md (summary tables).
Note: agent runs made during evaluation are stored like any other run.
"""

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI

from app.application.dto.pagination import PageRequest
from app.application.prompts import agent as agent_prompts
from app.application.prompts import answer as answer_prompts
from app.bootstrap.container import Container
from app.core.config import get_settings
from evals.dataset import load_dataset
from evals.judge import Judge
from evals.report import answer_summary, markdown, retrieval_summary
from evals.runner import RETRIEVAL_CONFIGS, SYSTEMS, evaluate_answers, evaluate_retrieval

ROOT = Path(__file__).resolve().parent


async def main(args: argparse.Namespace) -> None:
    settings = get_settings()
    container = Container(settings)
    judge_client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        timeout=settings.llm_timeout_seconds,
    )
    judge = Judge(
        judge_client,
        args.judge_model or settings.llm_model,
        reasoning_effort=args.judge_effort or None,
    )
    try:
        collections = await container.list_collections().execute(PageRequest(limit=100))
        collection = next((c for c in collections if c.name.value == args.collection), None)
        if collection is None:
            raise SystemExit(f"No collection named '{args.collection}' (run make ingest-samples)")
        questions = load_dataset(args.dataset)[: args.limit]
        print(
            f"{len(questions)} questions from {args.dataset.name}, collection '{collection.name}'"
        )

        retrieval_rows = []
        if not args.skip_retrieval:
            configs = [c for c in RETRIEVAL_CONFIGS if c.rerank is False or container.reranker]
            retrieval_rows = await evaluate_retrieval(container, collection.id, questions, configs)
        answer_rows = []
        if not args.skip_answers:
            systems = [s.strip() for s in args.systems.split(",")]
            answer_rows = await evaluate_answers(
                container, collection.id, questions, systems, judge
            )
    finally:
        await container.aclose()
        await judge_client.close()

    finished = datetime.now(UTC)
    meta = {
        "dataset": args.dataset.name,
        "questions": len(questions),
        "collection": args.collection,
        "embedding_model": settings.embedding_model,
        "chat_model": settings.llm_model,
        "judge_model": judge.model,
        "reranker": container.reranker.name if container.reranker else None,
        "prompts": {"answer": answer_prompts.PROMPT_VERSION, "agent": agent_prompts.PROMPT_VERSION},
        "settings": {
            "rerank_depth": settings.rerank_depth,
            "chunk_target_tokens": settings.chunk_target_tokens,
            "answer_context_tokens": settings.answer_context_tokens,
            "agent_max_tool_calls": settings.agent_max_tool_calls,
        },
        "finished_at": finished.strftime("%Y-%m-%d %H:%M UTC"),
    }
    retrieval = retrieval_summary(retrieval_rows)
    answers = answer_summary(answer_rows)
    report = markdown(retrieval, answers, meta)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / finished.strftime("%Y%m%d-%H%M%S")
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "meta": meta,
                "retrieval_summary": retrieval,
                "answer_summary": answers,
                "retrieval": [asdict(r) for r in retrieval_rows],
                "answers": [r.as_dict() for r in answer_rows],
            },
            indent=2,
            default=str,
        )
    )
    stem.with_suffix(".md").write_text(report)
    print("\n" + report)
    print(f"Saved {stem}.json and .md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RAG evaluation")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "samples.jsonl")
    parser.add_argument("--collection", default="samples")
    parser.add_argument("--systems", default=",".join(SYSTEMS), help="ask,agent")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-answers", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="first N questions only")
    parser.add_argument("--judge-model", default=None, help="default: LLM_MODEL")
    parser.add_argument("--judge-effort", default="low", help="reasoning effort for the judge")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    asyncio.run(main(parser.parse_args()))
