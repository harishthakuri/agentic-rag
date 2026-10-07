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

| Layer | Responsibility |
|---|---|
| `app/domain` | Entities, value objects, repository interfaces, domain errors |
| `app/application` | Use cases and ports (LLM, embeddings, reranker, unit of work, ...) |
| `app/infrastructure` | Adapters: SQLAlchemy/pgvector, OpenAI-compatible clients, parsers, worker |
| `app/presentation` | FastAPI routers, request/response schemas, middleware |
| `app/bootstrap` | Composition root that wires adapters to ports |

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

# 3. Dependencies, git hooks and migrations
make install
make migrate

# 4. Run
make run                    # http://localhost:8000/docs
```

## Development

```bash
make check             # lint + typecheck + architecture contracts + tests
make test-integration  # tests against a real pgvector container
make help              # list all commands
```
