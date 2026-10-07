# Simple RAG

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

## Using the API

Every `/api/v1` route requires an API key. Only its hash is stored, so the key is shown once:

```bash
make api-key name=dev        # prints srag_...
export RAG_KEY=srag_...

curl -X POST localhost:8000/api/v1/collections \
  -H "Authorization: Bearer $RAG_KEY" -H "Content-Type: application/json" \
  -d '{"name": "kubernetes-docs", "description": "K8s notes"}'

uv run python -m app.presentation.cli api-key list            # list keys
uv run python -m app.presentation.cli api-key revoke <id>     # revoke a key
```

### Ingest documents

Uploads return `202 Accepted` immediately. The **worker** parses, chunks and embeds in the background:

```bash
make worker                                   # in a second terminal
RAG_API_KEY=srag_... make ingest-samples      # uploads sample_data/ into collection "samples"
```

Or upload your own (`.md`, `.txt`, `.pdf`):

```bash
curl -X POST localhost:8000/api/v1/collections/<collection-id>/documents \
  -H "Authorization: Bearer $RAG_KEY" -F "file=@notes.md"
# → follow the returned job: GET /api/v1/jobs/<job-id>
```

Errors use [RFC 9457 Problem Details](https://www.rfc-editor.org/rfc/rfc9457) (`application/problem+json`).

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
Authorization: Bearer srag_xxxxxxxx...
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
| `uv run python -m app.presentation.cli api-key list`        | List keys (name, prefix, last used, status) |
| `uv run python -m app.presentation.cli api-key revoke <id>` | Revoke a key, e.g. if it leaked             |

Give each client its own key: one for you, one for a script, and so on. Then you can revoke one without breaking the others.

## What to try now in Swagger ([http://localhost:8000/docs](http://localhost:8000/docs))

1. **`POST /api/v1/collections`** with `{"name": "kubernetes-docs", "description": "my notes"}`. A collection is a named bucket of documents that you search together. The response shows it's bound to `qwen3-embedding:8b` at 1024 dims.
2. **`GET /api/v1/collections`** to list it.
3. **`POST`** the same name again. You get **409 Conflict** in the standard error format.
4. **`GET /api/v1/collections/{id}/documents`** returns an empty list, because there's no way to upload documents yet.

Right now you can manage collections, but there's nothing to search. The actual RAG part starts in Phase 3.
