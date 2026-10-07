# Agentic RAG

Retrieval-Augmented Generation with **hybrid search**, **reranking** and **agentic search**, built on **Clean Architecture**.

- **Stack:** FastAPI · PostgreSQL + pgvector · SQLAlchemy (async) · Alembic · OpenAI SDK (Ollama or OpenAI) · uv
- **Models (local by default):** `qwen3-embedding:8b` for embeddings and `gpt-oss:20b` for answering and agent reasoning, both through Ollama. You can switch to OpenAI with three environment variables.

> 🚧 Work in progress. See the [implementation plan](docs/IMPLEMENTATION_PLAN.md) for the design and roadmap.

## Architecture

```
presentation  →  bootstrap (composition root)  →  infrastructure  →  application  →  domain
```

Dependencies point inwards only. `domain` and `application` contain no framework code. The rule is enforced by [import-linter](https://github.com/seddonym/import-linter) (`make arch`).

| Layer                | Responsibility                                                            |
| -------------------- | ------------------------------------------------------------------------- |
| `app/domain`         | Entities, value objects, repository interfaces, domain errors             |
| `app/application`    | Use cases and ports (LLM, embeddings, reranker, unit of work, ...)        |
| `app/infrastructure` | Adapters: SQLAlchemy/pgvector, OpenAI-compatible clients, parsers, worker |
| `app/presentation`   | FastAPI routers, request/response schemas, middleware                     |
| `app/bootstrap`      | Composition root that wires adapters to ports                             |

## Getting started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.13
- PostgreSQL 16+ with the [pgvector](https://github.com/pgvector/pgvector) extension
- [Ollama](https://ollama.com) with the models pulled:
  ```bash
  ollama pull qwen3-embedding:8b
  ollama pull gpt-oss:20b
  ```
- Docker, for integration tests only

### Setup

```bash
# 1. Database roles: run once as a PostgreSQL superuser (see instructions in the file)
psql -h <host> -U postgres -d simple-rag-db -f scripts/db/001_roles.sql

# 2. Configuration
cp .env.example .env        # fill in the passwords you chose in step 1

# 2.1 Verify the database connection
uv run python -c "import asyncio; from app.core.config import get_settings; from app.bootstrap.container import Container; c=Container(get_settings()); print(asyncio.run(c.check_readiness().execute()))"

# 3. Dependencies, git hooks and migrations
make install
make migrate

# 4. Run
make run                    # http://localhost:8000/docs
```

## Web UI

Open **http://localhost:8000** (it redirects to `/ui/`) and paste an API key in the sidebar. The UI is a single static page that uses the same public API as any other client:

- **Collections and documents:** create collections, upload Markdown, text or PDF files, and watch ingestion progress live.
- **Chat:** streamed answers in **Ask** mode (one search) or **Agent** mode (the model plans its own searches, shown step by step). Citations `[n]` are clickable and open the exact source passage.
- **Search lab:** one query through vector, keyword, hybrid and hybrid + rerank, side by side. Hover a result to see where the same passage ranks in the other columns.

There is no build step: plain HTML, CSS and JavaScript, with two vendored libraries (Markdown rendering and HTML sanitising). It is served under a strict Content-Security-Policy, and model output is always sanitised before it is displayed.

## Using the API

Every `/api/v1` route requires an API key. Only its hash is stored, so the key is shown once:

```bash
make api-key name=dev        # prints arag_...
export RAG_KEY=arag_...

curl -X POST localhost:8000/api/v1/collections \
  -H "Authorization: Bearer $RAG_KEY" -H "Content-Type: application/json" \
  -d '{"name": "kubernetes-docs", "description": "K8s notes"}'

uv run rag-admin api-key list            # list keys
uv run rag-admin api-key revoke <id>     # revoke a key
```

### Ingest documents

Uploads return `202 Accepted` immediately. The **worker** parses, chunks and embeds in the background:

```bash
make worker                                   # in a second terminal
RAG_API_KEY=arag_... make ingest-samples      # uploads sample_data/ into collection "samples"
```

Or upload your own (`.md`, `.txt`, `.pdf`):

```bash
curl -X POST localhost:8000/api/v1/collections/<collection-id>/documents \
  -H "Authorization: Bearer $RAG_KEY" -F "file=@notes.md"
# → follow the returned job: GET /api/v1/jobs/<job-id>
```

### Search

```bash
curl -X POST localhost:8000/api/v1/collections/<collection-id>/search \
  -H "Authorization: Bearer $RAG_KEY" -H "Content-Type: application/json" \
  -d '{"query": "Why does my filtered vector search return fewer rows?", "mode": "hybrid"}'
```

`mode` is `vector` (semantic), `keyword` (full-text) or `hybrid` (both, fused with Reciprocal Rank Fusion). Each hit shows its rank in both retrievers.

With `RERANKER=llm` (the default), the top candidates are then **reranked** by the chat model, which grades each passage 0–3 for how well it answers the query. This is more precise but takes seconds. Pass `"rerank": false` to skip it. To compare all modes side by side:

```bash
uv run python scripts/compare_search.py "your question"
```

### Ask (RAG)

```bash
curl -N -X POST localhost:8000/api/v1/collections/<collection-id>/ask \
  -H "Authorization: Bearer $RAG_KEY" -H "Content-Type: application/json" \
  -d '{"question": "Why does my filtered vector search return fewer rows than the LIMIT?", "stream": true}'
```

Search (hybrid + rerank) runs once with the question. The chat model then answers **only from the retrieved sources**, citing them as `[n]`. With `"stream": true` you get Server-Sent Events (`sources`, then `token`…, then `done`). Without it, you get a single JSON answer. Every response lists the sources and which ones were cited. If nothing relevant is found, the model is not called at all.

### Agentic search

```bash
curl -N -X POST localhost:8000/api/v1/collections/<collection-id>/agent/ask \
  -H "Authorization: Bearer $RAG_KEY" -H "Content-Type: application/json" \
  -d '{"question": "My cluster is bare-metal: how do users reach my HTTPS API, and how do I stop browsers using stale JavaScript?", "stream": true}'
```

The model decides **what to search for, how often, and when to stop**. It can split a question into several searches, rephrase, and read around a passage, then answer with citations. The stream shows each step (`tool_call`, `tool_result`) as it happens. Every run is stored and can be replayed with `GET /api/v1/agent/runs/<run-id>`.

Errors use [RFC 9457 Problem Details](https://www.rfc-editor.org/rfc/rfc9457) (`application/problem+json`).

## Evaluation

A labelled question set measures retrieval (recall, MRR, nDCG) and answer quality (correctness, faithfulness and citations, graded by an LLM judge). [Method, results and their interpretation](docs/EVALUATION.md).

| System | Correctness | Faithfulness | Answers with citations | Declined unanswerable | p50 latency |
|---|---|---|---|---|---|
| `/ask` (hybrid + LLM rerank) | 1.00 | 0.97 | 79% | 3/3 | 7.6 s |
| `/agent/ask` | 1.00 | 0.95 | 88% | 3/3 | 6.3 s |

The main findings: the sample corpus is too easy to separate the retrieval strategies, and the remaining weakness is citation discipline, not retrieval. Building the evaluation also caught three bugs, two of them in production code.

```bash
make eval-retrieval    # ~3 min
make eval              # ~30 min locally (about 250 LLM calls)
```

## Run with Docker

The image runs in three roles: API, ingestion worker and migrations. `docker compose` adds PostgreSQL + pgvector, provisioned with the same least-privilege roles as production:

```bash
docker compose up -d --build                         # or: make up
docker compose exec api rag-admin api-key create --name dev
open http://localhost:8000
```

Ollama stays on the host (Docker on macOS can't use the Apple GPU). Set `OLLAMA_URL` for your Docker runtime:

| Runtime | `OLLAMA_URL` |
|---|---|
| Docker Desktop | `http://host.docker.internal:11434` (default) |
| Rancher Desktop | `http://host.lima.internal:11434` |
| Linux | default, with Ollama started using `OLLAMA_HOST=0.0.0.0` |

The image is about 80 MB on a slim Python base. It runs as a non-root user, with the tokenizer data downloaded at build time and JSON logs. Database passwords in `docker-compose.yml` are local development defaults; override them through environment variables.

## Development

```bash
make check             # lint + typecheck + architecture contracts + tests
make test-integration  # tests against a real pgvector container
uv run pytest -m live  # opt-in checks against your local Ollama models
make help              # list all commands
```

## What the API key is

It's a password for programs instead of people. Any client calling `/api/v1/...` (Swagger, curl, a script, later a UI) has to send it in a header:

```
Authorization: Bearer arag_xxxxxxxx...
```

Without a valid key, the API returns **401 Unauthorized**. Clicking **Authorize** in Swagger just makes Swagger add that header to every request for you.

How it works:

1. `make api-key name=dev` generates a random secret, like a 43-character password, and prints it **once**.
2. The database stores only a **SHA-256 hash** of it, never the key itself. If someone stole a copy of the database, they still couldn't use your key.
3. On each request, the API hashes the key you sent and looks up the hash. A match on a key that isn't revoked lets the request through.
4. `/health/live` and `/health/ready` don't need a key.

| Command                                                     | What it does                                |
| ----------------------------------------------------------- | ------------------------------------------- |
| `make api-key name=dev`                                     | Create a key                                |
| `uv run rag-admin api-key list`        | List keys (name, prefix, last used, status) |
| `uv run rag-admin api-key revoke <id>` | Revoke a key, e.g. if it leaked             |

Give each client its own key: one for you, one for a script, and so on. Then you can revoke one without breaking the others.

## What to try now in Swagger ([http://localhost:8000/docs](http://localhost:8000/docs))

1. **`POST /api/v1/collections`** with `{"name": "kubernetes-docs", "description": "my notes"}`. A collection is a named bucket of documents that you search together. The response shows it's bound to `qwen3-embedding:8b` at 1024 dims.
2. **`GET /api/v1/collections`** to list it.
3. **`POST`** the same name again. You get **409 Conflict** in the standard error format.
4. **`GET /api/v1/collections/{id}/documents`** returns an empty list, because there's no way to upload documents yet.

Right now you can manage collections, but there's nothing to search. The actual RAG part starts in Phase 3.
