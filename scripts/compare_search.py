"""Compare vector, keyword and hybrid search side by side on the same questions.

    uv run python scripts/compare_search.py "your question" ["another question" ...]
    uv run python scripts/compare_search.py --collection samples --top 3

Without questions, a built-in set illustrating each method's strengths is used.
Talks to the database and embedding model directly (no API server needed).
"""

import argparse
import asyncio

from app.application.dto.pagination import PageRequest
from app.application.use_cases.retrieval import SearchMode, SearchQuery
from app.bootstrap.container import Container
from app.core.config import get_settings

DEFAULT_QUESTIONS = [
    # Paraphrase: shares almost no words with the answer ("Cache busting").
    "How do I make sure a CDN never serves an old version of my JavaScript file?",
    # Exact identifiers: keyword search's home turf.
    "ef_search",
    "What does a 304 response mean?",
    # Conceptual question with some keyword overlap: hybrid should combine both.
    "Why does my filtered vector search return fewer rows than the LIMIT?",
    "How do I expose services on a bare-metal cluster without a cloud load balancer?",
]


async def main(collection_name: str, questions: list[str], top: int) -> None:
    container = Container(get_settings())
    try:
        collections = await container.list_collections().execute(PageRequest(limit=100))
        collection = next((c for c in collections if c.name.value == collection_name), None)
        if collection is None:
            raise SystemExit(f"No collection named '{collection_name}'")

        search = container.search_collection()
        variants = [(mode.value, mode, False) for mode in SearchMode]
        if container.reranker is not None:
            variants.append(("hybrid+rerank", SearchMode.HYBRID, True))

        for question in questions:
            print(f"\n\033[1mQ: {question}\033[0m")
            for label, mode, rerank in variants:
                result = await search.execute(
                    SearchQuery(collection.id, question, mode=mode, top_k=top, rerank=rerank)
                )
                note = f"  ! {result.rerank_error}" if result.rerank_error else ""
                print(f"  {label:13} ({result.timings_ms['total']:7.1f} ms){note}")
                if not result.hits:
                    print("                (no results)")
                for i, hit in enumerate(result.hits, start=1):
                    m = hit.match
                    where = " > ".join((m.document_title, *m.heading_path))
                    if rerank:
                        detail = f"  [grade {hit.rerank_score:.0f}, was #{hit.retrieval_rank}]"
                    elif mode is SearchMode.HYBRID:
                        v, k = hit.vector_rank or "-", hit.keyword_rank or "-"
                        detail = f"  [vector #{v}, keyword #{k}]"
                    else:
                        detail = ""
                    print(f"     {i}. {hit.score:8.4f}  {where}{detail}")
    finally:
        await container.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare search modes")
    parser.add_argument("questions", nargs="*")
    parser.add_argument("--collection", default="samples")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main(args.collection, args.questions or DEFAULT_QUESTIONS, args.top))
