# Simple RAG: Implementation Plan

> Status: **In progress.** Phases 0–7 are complete.
> Last updated: 2026-10-07

---

## 1. Purpose

This project builds a **Retrieval-Augmented Generation (RAG)** service with **agentic search**, step by step. The goals are to:

1. **Learn how RAG works:** embeddings, vector similarity, chunking, keyword vs semantic search, hybrid fusion, reranking, and LLMs that call retrieval as a tool.
2. **Build it the way a production system is built:** Clean Architecture, explicit boundaries, tests, migrations, least-privilege database access, observability, and measurable quality.
3. **Measure instead of guess:** an evaluation harness compares retrieval strategies (vector vs hybrid vs hybrid + rerank vs agentic) with real numbers.

### What we are building

A **knowledge-base API**. You upload Markdown and PDF documents into named **collections**, and then:

- **`/search`**: hybrid retrieval only. Returns ranked chunks with scores, so you can see what retrieval does on its own.
- **`/ask`**: classic one-shot RAG (retrieve → rerank → generate an answer with citations).
- **`/agent/ask`**: agentic search. The LLM decides *whether* and *what* to search, can refine its queries, read neighbouring context and search again, and then answers with citations. Every step is streamed (SSE) and stored, so you can watch it think.

### Non-goals for v1

- A web UI. We use Swagger at `/docs` plus `curl`/HTTPie. A UI can come later.
- Multi-tenant user management. A single owner with API keys is enough.
- Fine-tuning models.
- GraphRAG and knowledge graphs (listed under Future Work).

---

## 2. Tech Stack

| Concern | Choice | Why |
|---|---|---|
| Language | **Python 3.13** | Mature library compatibility across the AI and DB stack |
| Package manager | **uv** | Fast, lockfile-based, reproducible |
| Web framework | **FastAPI** | Async, typed, OpenAPI out of the box, SSE streaming |
| Settings | **pydantic-settings** | Typed, validated config from `.env` and the environment, no module-level `os.getenv` |
| Database | **PostgreSQL 16** (existing server, `simple-rag-db`) | One store for relational data, vectors *and* full-text search |
| Vector search | **pgvector 0.8.2** (`vector`, HNSW index) | Approximate nearest-neighbour search inside Postgres; 0.8 adds *iterative index scans* for filtered queries |
| Keyword search | **Postgres full-text search** (`tsvector` + GIN, `ts_rank_cd`) | BM25-like lexical relevance with no extra infrastructure |
| ORM | **SQLAlchemy 2.x (async)** + **asyncpg** | Typed async ORM; raw SQL where it is clearer (hybrid search) |
| Migrations | **Alembic** (async env) | Versioned, reviewable schema changes |
| Embeddings | **Ollama → `qwen3-embedding:8b`**, truncated to **1024 dims** | Strong multilingual embedding model running locally (see §4.1) |
| Chat / agent LLM | **Ollama → `gpt-oss:20b`** by default; any OpenAI model via config | Local, free and private by default; switch providers with environment variables |
| LLM client | **OpenAI Python SDK** | One SDK for both Ollama (OpenAI-compatible API) and OpenAI, so the provider is just `base_url` + `api_key` |
| Reranker | **LLM-based reranker** (v1), **cross-encoder** (later) | Behind a `Reranker` port (see §5) |
| Document parsing | Small line-based Markdown parser, `pypdf` (PDF) | Heading-aware (code-fence safe), lightweight, permissively licensed |
| Token counting | `tiktoken` | Chunk sizing in tokens rather than characters |
| Background jobs | **Postgres-backed job queue** (`FOR UPDATE SKIP LOCKED`) + a worker process | Reliable async ingestion without Redis or Celery |
| Logging | `structlog` (JSON in prod, pretty in dev) | Structured logs with request ID and per-stage timings |
| Testing | `pytest`, `pytest-asyncio`, `httpx`, **testcontainers** (`pgvector/pgvector:pg16`) | Real-database integration tests; fakes for the LLM and embeddings |
| Code quality | `ruff` (lint + format), `mypy` (strict on domain/application), `pre-commit` | Consistent, type-safe code |
| Container runtime (dev/test) | Docker (Rancher Desktop) | testcontainers only; the app talks to the existing Postgres server |

---

## 3. Architecture

### 3.1 Clean Architecture layers

The layout follows the reference project (`fastapi-clean-architecture`): `domain` / `application` / `infrastructure` / `presentation`. It adds a real **application layer** (use cases), a **Unit of Work**, and a single **composition root**.

```
            ┌──────────────────────────────────────────────┐
            │  presentation  (FastAPI routers, schemas,    │
            │                 SSE, auth, error mapping)    │
            └───────────────────────┬──────────────────────┘
                                    │ calls
            ┌───────────────────────▼──────────────────────┐
            │  application   (use cases, DTOs, ports for   │
            │                 LLM / embedder / reranker /  │
            │                 parser / chunker / UoW)      │
            └───────────────────────┬──────────────────────┘
                                    │ depends on
            ┌───────────────────────▼──────────────────────┐
            │  domain        (entities, value objects,     │
            │                 repository interfaces,       │
            │                 domain errors). Pure Python, │
            │                 no frameworks.               │
            └──────────────────────────────────────────────┘
                                    ▲ implements interfaces
            ┌───────────────────────┴──────────────────────┐
            │  infrastructure (SQLAlchemy repos, pgvector  │
            │                  search, OpenAI/Ollama       │
            │                  adapters, parsers, worker)  │
            └──────────────────────────────────────────────┘
```

**Dependency rule:** source dependencies point inwards only. `domain` imports nothing from the other layers. `application` imports only `domain`. `infrastructure` and `presentation` are the outer ring. Only the composition root (`app/bootstrap/container.py`) knows about every concrete class. This rule will be **enforced in CI** with `import-linter`.

### 3.2 Ports (interfaces) and adapters

| Port (application/domain) | v1 adapter (infrastructure) | Swappable to |
|---|---|---|
| `EmbeddingProvider` | `OpenAICompatibleEmbedder` (Ollama `qwen3-embedding:8b`) | OpenAI `text-embedding-3-*`, any OpenAI-compatible server |
| `ChatModel` | `OpenAICompatibleChatModel` (Ollama `gpt-oss:20b`) | Any OpenAI model, vLLM, LM Studio |
| `Reranker` | `LLMReranker`, `NoopReranker` | `CrossEncoderReranker` (local), hosted rerank APIs |
| `DocumentParser` | `MarkdownParser`, `PdfParser` | Docling, Unstructured, OCR |
| `Chunker` | `StructureAwareChunker` | Semantic chunker, late chunking |
| `ChunkSearchRepository` | `PgHybridSearchRepository` | Qdrant, OpenSearch |
| `CollectionRepository`, `DocumentRepository`, `JobRepository`, ... | SQLAlchemy implementations | — |
| `UnitOfWork` | `SqlAlchemyUnitOfWork` | — |

> Switching from local Ollama to OpenAI is configuration only:
> `LLM_BASE_URL=https://api.openai.com/v1`, `LLM_API_KEY=sk-...`, `LLM_MODEL=<model>`.
> No code changes. Embeddings are configured separately (`EMBEDDING_*`), because chat and embedding providers are often different.

### 3.3 Project structure

```
simple-rag/
├── app/
│   ├── domain/
│   │   ├── models/            # Collection, Document, Chunk, IngestionJob, AgentRun, ...
│   │   ├── value_objects/     # Embedding, ChunkText, ContentHash, Citation, ...
│   │   ├── repositories/      # Abstract repository interfaces
│   │   └── exceptions/
│   ├── application/
│   │   ├── ports/             # EmbeddingProvider, ChatModel, Reranker, Parser, Chunker, UoW
│   │   ├── use_cases/
│   │   │   ├── collections/   # create / list / delete
│   │   │   ├── ingestion/     # upload_document, process_ingestion_job
│   │   │   ├── retrieval/     # hybrid_search
│   │   │   ├── answering/     # ask (one-shot RAG)
│   │   │   └── agent/         # agentic_ask (tool loop), tool definitions
│   │   ├── dto/
│   │   └── prompts/           # Versioned prompt templates (plain text files)
│   ├── infrastructure/
│   │   ├── persistence/
│   │   │   ├── orm/           # SQLAlchemy entities
│   │   │   ├── repositories/  # Repository implementations + mappers
│   │   │   ├── search/        # Hybrid search SQL (RRF)
│   │   │   ├── database.py    # Engine / session factory (no globals at import time)
│   │   │   └── unit_of_work.py
│   │   ├── llm/               # OpenAI-compatible chat + embedding adapters
│   │   ├── reranking/         # LLM / Noop / (CrossEncoder later)
│   │   ├── parsing/           # Markdown, PDF parsers
│   │   ├── chunking/
│   │   ├── jobs/              # Worker loop, job runner
│   │   └── security/          # API key hashing
│   ├── presentation/
│   │   └── api/
│   │       ├── v1/routers/    # collections, documents, search, ask, agent, jobs, health
│   │       ├── schemas/       # Request/response Pydantic models
│   │       ├── sse.py
│   │       ├── auth.py
│   │       └── error_handlers.py
│   ├── bootstrap/
│   │   └── container.py       # Composition root: wires ports → adapters
│   ├── core/
│   │   ├── config.py          # pydantic-settings
│   │   └── logging.py
│   ├── main.py                # FastAPI app factory
│   └── worker.py              # Ingestion worker entry point
├── alembic/                   # Migrations
├── evals/
│   ├── datasets/              # *.jsonl question sets
│   └── runner.py              # `uv run rag-eval`
├── scripts/
│   └── db/001_roles.sql       # Run once as superuser (instructions inside)
├── sample_data/               # Small corpus to ingest and evaluate against
├── tests/
│   ├── unit/
│   ├── integration/           # testcontainers + pgvector
│   └── e2e/
├── docs/
│   ├── IMPLEMENTATION_PLAN.md # (this file)
│   └── adr/                   # Architecture Decision Records
├── .env.example
├── alembic.ini
├── pyproject.toml
└── README.md
```

---

## 4. Core RAG Design Decisions

### 4.1 Embeddings: `qwen3-embedding:8b` at 1024 dimensions

- **The problem:** the model outputs **4096-dim** vectors. pgvector's HNSW index supports at most **2000 dims** for `vector` (4000 for `halfvec`). At 4096 dims there is no index, so every query scans the whole table.
- **The fix:** Qwen3-Embedding is trained with **Matryoshka Representation Learning (MRL)**, which puts the most important information in the first dimensions. We request `dimensions=1024` through Ollama's OpenAI-compatible `/v1/embeddings` endpoint (verified locally) and store `vector(1024)`.
- **Trade-off:** a small quality loss in exchange for an index, ~4× less storage, and faster search. The eval harness will measure 1024 vs 2048 (`halfvec`) so this is a measured decision, not a guess.
- **Asymmetric prompting:** Qwen3-Embedding expects **queries** to carry a task instruction, while **documents** are embedded as plain text:
  ```
  Instruct: Given a user question, retrieve passages from the knowledge base that answer it
  Query: <user question>
  ```
  The `EmbeddingProvider` port exposes `embed_query()` and `embed_documents()` separately, so this detail cannot be forgotten.
- **Model lock-in guard:** each collection records its `embedding_model` and `embedding_dim`. Mixing vectors from different models in one index produces meaningless similarity scores, so the API refuses it.
- **Distance metric:** cosine distance (`vector_cosine_ops`, `<=>`). Vectors are normalised.

### 4.2 Chunking: by document structure, not fixed size

- **Markdown:** split on the heading hierarchy (`#`, `##`, `###`). If a section is too long, split it on paragraphs, targeting **~500 tokens** with **~60 tokens of overlap**. Code blocks and tables are never split in the middle.
- **PDF:** extract text page by page (`pypdf`), then split on paragraphs with the same token targets. The page number is kept so citations can say *"p. 12"*.
- **Contextual chunk header:** before embedding, each chunk is prefixed with its context:
  ```
  Document: Kubernetes Networking Guide
  Section: Services > ClusterIP
  ---
  <chunk text>
  ```
  This is a cheap version of *contextual retrieval*. Without the header, a chunk that says "it defaults to port 80" has lost what "it" refers to. The original text is stored separately for display.
- **Idempotency:** documents are deduplicated by a SHA-256 content hash. Re-uploading the same file does not create duplicate chunks.

### 4.3 Hybrid search with Reciprocal Rank Fusion (RRF)

Vector search finds *meaning* ("car" ≈ "automobile") but misses exact tokens such as error codes, function names and acronyms. Keyword search is the opposite. Combining them is the sensible default.

**One SQL query** runs both searches and fuses the results:

```sql
WITH semantic AS (
  SELECT id, row_number() OVER (ORDER BY embedding <=> :query_vec) AS rank
  FROM rag.chunks
  WHERE collection_id = :collection_id
  ORDER BY embedding <=> :query_vec
  LIMIT :candidates                          -- e.g. 50
),
lexical AS (
  SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS rank
  FROM rag.chunks, websearch_to_tsquery('english', :query_text) q
  WHERE collection_id = :collection_id AND tsv @@ q
  ORDER BY ts_rank_cd(tsv, q) DESC
  LIMIT :candidates
)
SELECT id,
       COALESCE(1.0 / (:k + s.rank), 0) + COALESCE(1.0 / (:k + l.rank), 0) AS rrf_score
FROM semantic s FULL OUTER JOIN lexical l USING (id)
ORDER BY rrf_score DESC
LIMIT :top_n;                                -- k = 60 (standard RRF constant)
```

- **Why RRF:** cosine distances and `ts_rank` scores are on different scales and cannot be added directly. RRF uses **ranks only**, so no score calibration is needed.
- **Filtered vector search:** filtering by `collection_id` *after* an HNSW scan can return too few rows. pgvector 0.8's `SET LOCAL hnsw.iterative_scan = relaxed_order` keeps scanning until enough rows match. This is a subtle production issue, and we handle it explicitly.
- **Search modes** (`vector`, `keyword`, `hybrid`) can be selected per request, so you can compare them side by side.
- **As implemented:** the two retrievers are separate queries behind a `ChunkSearchIndex` port, and RRF is a pure function in the domain layer (`domain/services/rank_fusion.py`). This keeps fusion storage-independent and testable, and lets each hit report its rank in *both* lists.
- **Keyword semantics:** `websearch_to_tsquery` ANDs every word, so a natural-language question usually matches *nothing* (verified on the sample corpus). Questions therefore OR their stemmed terms and let `ts_rank_cd` reward chunks matching more of them. Explicit search syntax (`"phrase"`, `or`, `-term`) still goes through `websearch_to_tsquery`.
- **Observed on the sample corpus** (`scripts/compare_search.py`): vector search wins on paraphrases, keyword search on exact identifiers. Hybrid is right when the two agree, but it can demote a correct vector hit when keyword search favours a chunk that merely repeats query words ("What does a 304 response mean?"). That is the motivation for reranking (§5).

### 4.4 Retrieval pipeline

```
query ──► embed_query ──► hybrid search (50 candidates)
                              │
                              ▼
                        Reranker (top 50 → top 8)
                              │
                              ▼
                  context assembly (dedupe, merge adjacent chunks,
                  order by document position, token budget)
                              │
                              ▼
                     LLM answer with [n] citations
```

### 4.4.1 One-shot answering (`/ask`), as implemented

- **Context assembly**: chunks the reranker graded 0 are dropped, adjacent chunks are merged into one source (removing the paragraphs repeated by chunk overlap), sources are numbered by relevance, and they are added while they fit a token budget (`ANSWER_CONTEXT_TOKENS`, default 3000).
- **No relevant sources, no LLM call.** When every candidate is graded irrelevant, `/ask` returns a fixed "couldn't find" answer instead of letting the model improvise (verified with "What is the capital of France?").
- **Prompt**: a versioned system prompt (`answer-v2`). Sources are wrapped in `<source id=… location=…>` delimiters and marked as untrusted. Answers must cite `[n]` and must say when the sources don't contain the answer.
- **Citations**: `gpt-oss` cites with `【1】` (its training format) even when asked for `[1]`. A per-character mapping normalises them, and it also works on streamed tokens split mid-citation. Every response reports `cited` and `invalid_citations` (numbers that match no source).
- **Streaming (SSE)**: `sources` → `token`… → `done`. Retrieval runs *before* the response starts, so 404/503 remain real HTTP errors. Failures during generation arrive as an in-band `error` event.
- **Latency (local)**: about 7–12 s in total, of which search + rerank is about 4–6 s and generation about 3–5 s. The first token arrives about 1 s after search completes.

### 4.5 Agentic search (`/agent/ask`)

A hand-written **tool-calling loop**. It does not use LangChain or LangGraph, because the point is to understand and control each step, and frameworks tend to cut across the architecture's boundaries.

**Tools exposed to the LLM** (all read-only):

| Tool | Purpose |
|---|---|
| `search_knowledge_base(query, mode?, top_k?)` | Hybrid search + rerank. Returns chunk IDs, titles, snippets |
| `read_chunk_context(chunk_id, before?, after?)` | Reads neighbouring chunks when a snippet is cut off |
| `list_documents(filter?)` | Shows what is in the collection, so the agent can target its queries |

**Loop:**

1. The system prompt describes the tools, citation format and rules ("answer only from retrieved content; say when it is not found").
2. The LLM either calls tools or produces a final answer.
3. Tool calls run; their results are appended to the conversation as tool messages.
4. The loop repeats until the LLM gives a final answer **or** a guard stops it: `max_steps` (default 6), a token budget, or a wall-clock timeout.
5. Citations are **validated**: every `[n]` must refer to a chunk the agent actually retrieved. Unknown citations are flagged.

**Observability:** every run is stored (`agent_runs`, `agent_steps`) with the queries issued, the chunks returned, latency and token usage, and every step is streamed over SSE:

```
event: step        data: {"n":1,"type":"tool_call","tool":"search_knowledge_base","args":{...}}
event: step        data: {"n":1,"type":"tool_result","chunks":[...]}
event: token       data: {"text":"The ClusterIP service..."}
event: citations   data: [...]
event: done        data: {"steps":3,"latency_ms":4210,"usage":{...}}
```

**As implemented, and what we learned:**

- **Tools:** `search_knowledge_base` (hybrid search, no rerank by default, since the agent judges relevance itself), `read_more_context`, and `list_documents`. Every passage the agent sees gets a run-wide source number, so citations work exactly as in `/ask`.
- **Guards:** a tool-call limit, after which tools are withdrawn and the model is told to answer. The same happens when the prompt grows past a size estimate. There is also a wall-clock timeout. Bad arguments, unknown tools and repeated searches are returned to the model as text instead of failing the run.
- **Citations:** with the citation rule only in the system prompt, `gpt-oss` answered *without any citations*. Repeating a one-line reminder at the end of every tool result fixed it (prompt `agent-v2`). Rules belong where the model is looking.
- **Query decomposition:** asked in the system prompt to "search each part separately", the model still sent one long combined query. The same guidance in the *tool description* ("one topic per query") made it split a two-topic question into two focused searches.
- **Faithfulness is not solved by prompting alone:** the agent occasionally adds unsupported details from its own knowledge (e.g. "Layer 2 or BGP mode" for MetalLB). Measuring this is the job of the evaluation harness (§7).
- **Ollama context window:** it is a server setting and can't be set through the OpenAI API. If it were small, Ollama would silently drop the start of the prompt. Measured here: 128k for `gpt-oss:20b`, loaded through the desktop app.
- **Latency (local):** about 9–19 s for 1–2 searches plus the answer.

**Model note:** `gpt-oss:20b` supports native tool calling and adjustable reasoning effort. Ollama exposes both through the OpenAI-compatible Chat Completions API, so the same adapter works for OpenAI models.

**Security note:** retrieved document text is **untrusted input** and could contain prompt injection. Mitigations: all tools are read-only, retrieved content is wrapped in clearly delimited blocks, and the system prompt tells the model to treat it as data, not instructions.

---

## 5. Reranking in Detail

### 5.1 What a reranker is and why it matters

Retrieval has two competing goals:

- **Recall:** get every relevant chunk somewhere in the candidate list.
- **Precision:** put the *best* chunks at the very top, because only the top few fit in the LLM's context.

The first-stage retrievers (vector + keyword) are optimised for **speed over millions of rows**, so they are good at recall but rough on exact ordering. A reranker is a **slower, more accurate second stage** that re-scores a small candidate set (e.g. 50) and keeps only the best (e.g. 8).

```
millions of chunks ──(fast, approximate)──► 50 candidates ──(slow, precise)──► 8 chunks to the LLM
        first stage: HNSW + full-text                  second stage: reranker
```

### 5.2 Bi-encoders vs cross-encoders: the key idea

| | **Bi-encoder** (our embedding model) | **Cross-encoder** (reranker) |
|---|---|---|
| How it works | Encodes the query and the document **separately** into vectors, then compares them with cosine similarity | Reads the query **and** the document **together** in one pass and outputs a relevance score |
| Can precompute document side? | **Yes**: embed once at ingestion | **No**: must run for each (query, doc) pair at query time |
| Speed | Very fast (one vector comparison) | Slow (one full model pass per pair) |
| Accuracy | Good: the whole document is squeezed into one vector, so detail is lost | Better: the model sees the exact interaction between query words and document words |
| Use | First stage over the whole corpus | Second stage over ~20–100 candidates |

**Example:** for the query *"How do I **disable** TLS verification?"*, a bi-encoder may rank a chunk about *enabling* TLS highly, because the topic is nearly identical. A cross-encoder reads both texts together and notices the negation.

### 5.3 Reranker options

| Option | How | Pros | Cons |
|---|---|---|---|
| **No reranker** | Use RRF order directly | Zero latency, baseline for comparison | Lowest precision |
| **LLM reranker** (v1) | Ask the chat LLM to score each candidate's relevance (0–3), as a structured JSON output, in batches | No new dependencies; works with any OpenAI-compatible model; explainable | Slower than a dedicated model; costs tokens; the LLM was not trained for this specific task |
| **Local cross-encoder** (phase 9) | `sentence-transformers` + e.g. `BAAI/bge-reranker-v2-m3` or `Qwen3-Reranker`, running on Apple Silicon (MPS) | Purpose-built and the most accurate per millisecond; private | Adds PyTorch (~1–2 GB); Ollama does not serve rerankers well, so it runs in-process or as a sidecar |
| **Hosted rerank API** | Cohere / Voyage / Jina rerank endpoints | Excellent quality, no local compute | Data leaves your machine; adds a vendor |

### 5.4 v1 design

```python
class Reranker(Protocol):  # application/ports
    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RankedChunk]: ...
```

- **`LLMReranker`:** sends the query plus a batch of candidates (truncated snippets) and asks for `[{"id": ..., "score": 0-3, "reason": "..."}]` as a JSON schema response. Ties are broken by the original RRF rank. Batches are scored concurrently. If the output cannot be parsed, it **falls back to RRF order** instead of failing the request.
- **`NoopReranker`:** passes the RRF order through. Used as the baseline and in tests.
- Selected with `RERANKER=llm|none|cross_encoder`. It can also be overridden per request (`"rerank": false`) for side-by-side comparison.
- **Latency budget:** reranking is the most expensive retrieval step. The response includes timings per stage (`embed_ms`, `search_ms`, `rerank_ms`, `generate_ms`) so the cost is visible.

### 5.5 As implemented, and what we measured

- **No reranker = `RERANKER=none`**, rather than a `NoopReranker` class: search simply skips the stage. `rerank` can also be set per request.
- **The schema must say exactly what you want.** With a generic "array of grades" JSON schema, `gpt-oss:20b` answered `{"grades": []}` every time: valid JSON, zero grades. The schema now requires exactly *n* grades and restricts passage numbers to `1..n`. After that change, every run graded every passage correctly. Validation stays in place, and any failure falls back to the first-stage order with `rerank_error` set.
- **Quality on the sample corpus** (`scripts/compare_search.py`): reranking fixed the case hybrid search got wrong ("What does a 304 response mean?": ETag moved from #2 to #1, and "Freshness" dropped to grade 0). It kept the correct #1 everywhere else. Grades also separate clearly relevant (3) from irrelevant (0) chunks, a signal `/ask` can use to drop useless context.
- **Latency is the cost**: about 3.4 s for 10 candidates (one batch) and 7.8 s for 20, against about 0.2 s for hybrid search alone (`gpt-oss:20b`, `reasoning_effort=low`, M1 Max). Ollama processes the batches sequentially, so concurrency doesn't help locally. `RERANK_DEPTH` defaults to 10. A cross-encoder (phase 9) should bring this down to about 0.1–0.3 s.

### 5.6 How we will judge it

The eval harness (§7) reports **recall@k, MRR and nDCG@k** for: `vector` → `hybrid` → `hybrid + LLM rerank` → later `hybrid + cross-encoder`. A reranker is only worth its latency if it measurably moves those numbers on our dataset. That comparison table goes in the README.

---

## 6. Data Model (schema `rag`)

| Table | Key columns |
|---|---|
| `collections` | `id (uuid)`, `name (unique)`, `description`, `embedding_model`, `embedding_dim`, `created_at` |
| `documents` | `id`, `collection_id → collections`, `title`, `source_uri`, `mime_type`, `content_hash (unique per collection)`, `status (pending/processing/ready/failed)`, `error`, `metadata jsonb`, timestamps |
| `chunks` | `id`, `document_id → documents (cascade)`, `collection_id` (denormalised for filtering), `ordinal`, `text`, `contextual_text`, `heading_path text[]`, `page`, `token_count`, `embedding vector(1024)`, `tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', contextual_text)) STORED`, `metadata jsonb` |
| `ingestion_jobs` | `id`, `document_id`, `status`, `attempts`, `last_error`, `run_after`, `locked_at`, `locked_by`, timestamps |
| `agent_runs` | `id`, `collection_id`, `question`, `answer`, `model`, `status`, `steps_count`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `created_at` |
| `agent_steps` | `id`, `run_id → agent_runs`, `n`, `type (tool_call/tool_result/answer)`, `payload jsonb`, `latency_ms` |
| `api_keys` | `id`, `name`, `key_prefix`, `key_hash (sha256)`, `created_at`, `last_used_at`, `revoked_at` |

**Indexes:**

- `chunks.embedding`: `HNSW (vector_cosine_ops) WITH (m = 16, ef_construction = 64)`
- `chunks.tsv`: `GIN`
- `chunks(collection_id)`, `chunks(document_id, ordinal)`
- `ingestion_jobs(status, run_after)`: partial index on runnable jobs

Uploaded raw files are stored on local disk (`STORAGE_DIR`) behind a `FileStorage` port. The database stores metadata only.

---

## 7. Evaluation Harness

- **Dataset:** `evals/datasets/*.jsonl`, one line per question:
  `{"question": "...", "expected_chunk_ids|expected_doc_titles": [...], "reference_answer": "..."}`
  We bootstrap it by having the LLM draft questions from sample chunks, then **review them by hand**.
- **Retrieval metrics:** recall@5/@10, MRR, nDCG@10, latency p50/p95.
- **Answer metrics:**
  - **faithfulness:** an LLM judge checks that every claim is supported by the cited chunks
  - **citation precision**
  - **answer correctness** against the reference answer
- **Run:** `uv run rag-eval --dataset evals/datasets/sample.jsonl --modes vector,hybrid,hybrid_rerank,agent`
- **Output:** a Markdown table plus JSON in `evals/results/`. The best results are summarised in the README.

---

## 8. Security, Configuration and Operations

### 8.1 Database roles (least privilege)

`scripts/db/001_roles.sql` is run **once by you** as a superuser. Instructions are in the file header.

| Role | Can | Used by |
|---|---|---|
| `simple_rag_owner` | Owns schema `rag`; runs DDL | Alembic migrations only |
| `simple_rag_app` | `SELECT/INSERT/UPDATE/DELETE` on `rag.*`, use sequences; **no DDL** | FastAPI app + worker |

- `ALTER DEFAULT PRIVILEGES FOR ROLE simple_rag_owner IN SCHEMA rag GRANT ...` so tables created by future migrations are automatically available to the app role.
- `REVOKE ALL ON SCHEMA public FROM PUBLIC` and `REVOKE CREATE ON DATABASE` for the app role.
- The `vector` extension is already installed (it requires a superuser to install).
- Two separate connection settings: `DATABASE_URL` (app) and `MIGRATIONS_DATABASE_URL` (owner).

### 8.2 Configuration (`.env`)

```
APP_ENV=dev
API_KEYS_ENABLED=true

DATABASE_URL=postgresql+asyncpg://simple_rag_app:***@10.10.10.5:5432/simple-rag-db
MIGRATIONS_DATABASE_URL=postgresql+asyncpg://simple_rag_owner:***@10.10.10.5:5432/simple-rag-db

LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=gpt-oss:20b

EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_API_KEY=ollama
EMBEDDING_MODEL=qwen3-embedding:8b
EMBEDDING_DIM=1024

RERANKER=llm            # llm | none | cross_encoder
STORAGE_DIR=./data/uploads
```

### 8.3 API security

- API keys are generated by a CLI command (`uv run rag-admin create-api-key --name dev`). Only the hash is stored, and the key is shown once.
- `Authorization: Bearer <key>` on every route except `/health`.
- Upload limits: file size, allowed MIME types, and sanitised filenames.
- CORS origins are configured, not `*`.

### 8.4 Observability

- `structlog` JSON logs with `request_id`, `collection_id`, and per-stage timings.
- `/health/live` (process is up) and `/health/ready` (DB + LLM endpoint + embedding endpoint reachable).
- Consistent error responses (RFC 9457 *Problem Details*), with domain errors mapped to HTTP status codes in one place.

---

## 9. API Surface (v1)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health/live`, `/health/ready` | Liveness / readiness |
| `POST` | `/api/v1/collections` | Create a collection |
| `GET` | `/api/v1/collections` | List collections |
| `DELETE` | `/api/v1/collections/{id}` | Delete a collection and all of its data |
| `POST` | `/api/v1/collections/{id}/documents` | Upload a file (multipart). Returns `202` + job ID |
| `GET` | `/api/v1/collections/{id}/documents` | List documents with status |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document and its chunks |
| `GET` | `/api/v1/jobs/{id}` | Ingestion job status |
| `POST` | `/api/v1/collections/{id}/search` | Retrieval only: `{query, mode, top_k, rerank}` |
| `POST` | `/api/v1/collections/{id}/ask` | One-shot RAG answer with citations (SSE or JSON) |
| `POST` | `/api/v1/collections/{id}/agent/ask` | Agentic search, streamed over SSE |
| `GET` | `/api/v1/agent/runs/{id}` | Full trace of a past agent run |

---

## 10. Implementation Steps

Each phase ends with passing tests and **one or more focused git commits**.

### Phase 0: Foundations
- [x] `git init`, `.gitignore`, `README.md` skeleton, this plan
- [x] `uv init` (Python 3.13), dependencies, `ruff`/`mypy`/`pytest` config, `pre-commit`
- [x] Folder layout per §3.3, `import-linter` contracts for the dependency rule
- [x] `core/config.py` (pydantic-settings), `core/logging.py`, FastAPI app factory, `/health/live`

### Phase 1: Database
- [x] `scripts/db/001_roles.sql` with run instructions. **You run it**, then fill in `.env`.
- [x] Async engine/session factory (`infrastructure/persistence/database.py`), `SqlAlchemyUnitOfWork`
- [x] Alembic async setup using `MIGRATIONS_DATABASE_URL`, `version_table_schema='rag'`
- [x] Migration 001: `collections`, `documents`, `chunks` (vector + tsvector + indexes), `ingestion_jobs`
- [x] testcontainers fixture (`pgvector/pgvector:pg16`) that runs migrations for integration tests
- [x] `/health/ready` checks the database

### Phase 2: Collections and documents (CRUD)
- [x] Domain entities, value objects and repository interfaces
- [x] SQLAlchemy repositories + mappers
- [x] Use cases + routers + schemas for collections and document listing/deletion
- [x] Problem Details error handling, API-key auth + `rag-admin` CLI

### Phase 3: Ingestion pipeline
- [x] `FileStorage` port + local disk adapter; upload endpoint (`202` + job)
- [x] `MarkdownParser`, `PdfParser` → a normalised `ParsedDocument` (sections, pages)
- [x] `StructureAwareChunker` with contextual headers + token counting (thorough unit tests)
- [x] `OpenAICompatibleEmbedder` (`dimensions=1024`, batching, retries, query instruction)
- [x] Postgres job queue + `app/worker.py` (`SKIP LOCKED`, retries with backoff, crash-safe)
- [x] Sample corpus in `sample_data/` and an ingest script

### Phase 4: Retrieval
- [x] `PgHybridSearchRepository`: `vector`, `keyword`, `hybrid` (RRF) modes with iterative scan
- [x] `/search` endpoint returning per-stage scores and ranks (for learning and debugging)
- [x] Integration tests with fixed vectors so the ranking is deterministic

### Phase 5: Reranking
- [x] `Reranker` port, `NoopReranker`, `LLMReranker` (structured output, batching, fallback)
- [x] `rerank` flag on `/search`, stage timings

### Phase 6: One-shot RAG (`/ask`)
- [x] `OpenAICompatibleChatModel` (streaming, tool calls, usage capture)
- [x] Context assembly (dedupe, merge neighbouring chunks, token budget)
- [x] Answer prompt with `[n]` citations, citation validation, SSE + JSON responses

### Phase 7: Agentic search (`/agent/ask`)
- [x] Tool schema definitions + handlers (`search_knowledge_base`, `read_chunk_context`, `list_documents`)
- [x] Agent loop with guards (max steps, token budget, timeout)
- [x] `agent_runs` / `agent_steps` migration + persistence; `/agent/runs/{id}`
- [x] SSE streaming of steps and tokens; e2e test with a scripted fake `ChatModel`

### Phase 8: Evaluation harness
- [ ] Dataset format + LLM-assisted question generation + manual review
- [ ] Retrieval and answer metrics, `rag-eval` CLI, Markdown/JSON reports
- [ ] Results table in the README: vector vs hybrid vs hybrid + rerank vs agent

### Phase 9: Polish and stretch goals
- [ ] `CrossEncoderReranker` (optional dependency group `[rerank-local]`) + eval comparison
- [ ] Embedding-dimension experiment: 1024 `vector` vs 2048 `halfvec`
- [ ] Dockerfile + `docker-compose.yml` for the app and worker
- [ ] GitHub Actions CI: ruff, mypy, import-linter, unit + integration tests
- [ ] ADRs in `docs/adr/` for the key decisions in this plan
- [ ] README with architecture diagram, quickstart and eval results

---

## 11. Testing Strategy

| Level | Scope | Tools |
|---|---|---|
| Unit | Domain rules, chunker, RRF math, citation validation, agent loop (fake LLM) | pytest, no I/O |
| Integration | Repositories, hybrid search SQL, migrations, job queue concurrency | testcontainers `pgvector/pgvector:pg16` |
| E2E | HTTP API through FastAPI with fake LLM/embedder adapters | httpx `AsyncClient` |
| Live (manual / opt-in) | Real Ollama models | `pytest -m live` |
| Quality | Retrieval and answer quality | Eval harness (§7) |

The fake `EmbeddingProvider` returns deterministic hash-based vectors, and the fake `ChatModel` replays scripted responses. Both are only possible because of the ports.

---

## 12. Future Work (beyond v1)

- **Query rewriting / HyDE / multi-query expansion** as agent tools
- **True BM25** via ParadeDB `pg_search`, measured against native full-text search
- **Late chunking** and **full contextual retrieval** (LLM-written chunk context)
- **GraphRAG** over document links (e.g. an Obsidian vault)
- **Conversation memory** for multi-turn agentic chat
- **OpenTelemetry tracing** of the LLM / tool / database spans
- A minimal web UI to visualise agent steps and retrieved chunks
- **MCP server** exposing `search_knowledge_base` to external agents

---

## 13. Risks and Open Questions

| Risk | Mitigation |
|---|---|
| `gpt-oss:20b` tool-calling quality or latency on an M1 Max | Model is configurable; the eval harness compares models; step limits bound latency |
| LLM reranker adds noticeable latency | Concurrent batches, snippet truncation, per-request opt-out, cross-encoder in phase 9 |
| Embedding model change invalidates stored vectors | Model + dim recorded per collection; re-embed job planned |
| PDF extraction quality (tables, multi-column layouts) | Parser behind a port; Docling is an upgrade path |
| Prompt injection via documents | Read-only tools, delimited context, explicit system instructions |
