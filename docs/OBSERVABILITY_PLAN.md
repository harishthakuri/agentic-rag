# Observability plan: OpenTelemetry, Grafana and Langfuse

> **Status: implemented** (all five steps of section 10). Section 14 lists where the
> implementation differs from the original plan and why.

This plan closes the gap "no metrics or tracing, only logs". It explains what OpenTelemetry is, what the app will record, what Grafana and Langfuse each show, how to configure and run them locally, and how the code changes are built.

---

## 1. Summary

**Today** the app writes structured logs (with a request ID) and returns `timings_ms` in each response. That is enough to debug one request by hand, but not to answer:

- Where did the 9 seconds of this slow answer go: search, reranker or the LLM?
- How often are answers withheld, repaired or "couldn't find" this week?
- How many tokens did we use today, and for which model?
- Is the reranker failing? Are uploads failing? How long does Docling take per document?

**After this work:**

| You get | Where | Example |
|---|---|---|
| A **trace** of every request and upload job: each step with its duration and details | Grafana (Tempo) and/or Langfuse | `ask` 6.8 s → embed 0.3 s → search 0.1 s → rerank 1.9 s → generate 4.4 s → citation check |
| **Metrics** over time, on a ready-made dashboard | Grafana (Prometheus) | p95 latency per step, tokens per hour, % answers withheld, failed uploads |
| **Logs linked to traces** | Grafana (Loki) | From an error log line, jump to the trace of that request |
| An **LLM view** of each question: prompts, answers, tokens, sources | Langfuse | Open a question, see the exact prompt gpt-oss received and the answer it wrote |

Everything is **off by default** and runs **locally in Docker**. No data leaves your machine, and no paid service is needed.

---

## 2. What is OpenTelemetry?

**OpenTelemetry (OTel)** is an open standard, with free open-source libraries, for recording what software does. It is a CNCF project (the foundation behind Kubernetes) and is supported by practically every monitoring tool: Grafana, Langfuse, Datadog, Honeycomb, New Relic, Elastic, AWS, Azure, Google Cloud.

It is **not a service and not a product**. It defines:

| Part | What it is |
|---|---|
| **Signals** | Three kinds of data: **traces**, **metrics** and **logs** (below) |
| **API + SDK** | Libraries the app uses to record those signals (Python packages, below) |
| **OTLP** | The wire protocol to send them anywhere (HTTP or gRPC) |
| **Semantic conventions** | Standard attribute names, e.g. `http.response.status_code`, `db.system`, and for AI calls `gen_ai.request.model`, `gen_ai.usage.input_tokens` |
| **Collector** (optional) | A relay that receives, processes and forwards telemetry. The Grafana container includes one. |

**The three signals, in our terms:**

- **Trace:** the story of one request. It is a tree of **spans**. A span is one step with a start, an end and attributes, e.g. "rerank 10 passages, 1.9 s, top score 0.999". All spans of one request share a **trace ID**.
- **Metric:** a number aggregated over time, e.g. a counter (answers withheld) or a histogram (LLM call duration, from which p50/p95 are computed). Cheap to keep for months; good for dashboards and alerts.
- **Log:** a timestamped event message, as today. With OTel, each log line also carries the trace ID of its request.

**Why OTel instead of a vendor SDK:** the app is instrumented once and the destination is configuration. The same code sends to Grafana today, to Langfuse or both, and to a cloud vendor later without code changes.

### Do we need third-party libraries or services?

**Libraries: yes, but only open-source OpenTelemetry packages** (Apache 2.0, maintained by the OpenTelemetry project):

| Package | Purpose |
|---|---|
| `opentelemetry-api`, `opentelemetry-sdk` | Record spans, metrics and logs; batch and export them |
| `opentelemetry-exporter-otlp-proto-http` | Send them over OTLP/HTTP (works for both Grafana and Langfuse) |
| `opentelemetry-instrumentation-fastapi` | Automatic span per HTTP request, plus request metrics |
| `opentelemetry-instrumentation-sqlalchemy` | Automatic span per database query |

They are small (no PyTorch). Proposal: add them to the **base dependencies**, so the Docker image has them; they do nothing until switched on.

Not used, on purpose: the Langfuse Python SDK (OTel covers it, and keeps us vendor-neutral), `opentelemetry-instrumentation-openai-v2` (still beta; we already know tokens and timings in our own adapters, so we record AI spans ourselves) and `opentelemetry-instrumentation-httpx` (the OpenAI SDK here is built on `httpx2`, which it doesn't see; our model-call spans cover those calls).

**Services: only where the data goes**, and both run locally in Docker (section 7):

- **Grafana LGTM**: one container, free and open source.
- **Langfuse**: six containers, free to self-host (core MIT licence).

You can use either, both or neither.

---

## 3. Architecture

```
 ┌──────────────── our app (no Docker needed) ────────────────┐
 │  API server  (service.name = agentic-rag-api)              │
 │  Worker      (service.name = agentic-rag-worker)           │
 │  Evals       (service.name = agentic-rag-evals)            │
 │                                                            │
 │  OpenTelemetry SDK: spans, metrics, logs                   │
 │     │ batched in the background, every ~5 s                │
 └─────┼──────────────────────────────────────────────────────┘
       │ OTLP/HTTP
       ├──────────────────────────────┐
       ▼ traces + metrics + logs      ▼ traces only
 ┌───────────────────────────┐  ┌──────────────────────────────┐
 │ Grafana LGTM  :4318       │  │ Langfuse  :3001/api/public/otel│
 │  Collector → Tempo        │  │  web + worker                │
 │            → Prometheus   │  │  Postgres, ClickHouse,       │
 │            → Loki         │  │  Redis, MinIO                │
 │  Grafana UI  :3000        │  │  UI  :3001                   │
 └───────────────────────────┘  └──────────────────────────────┘
```

The app exports directly to each destination you enable. If a destination is down, the export fails quietly in the background (logged as a warning) and **the app keeps working**.

---

## 4. What we will record

### 4.1 Traces

Automatic spans (from the instrumentation packages): every HTTP request and every SQL query. Calls to the models are recorded by our own adapters (below). On top of those, our own spans for each RAG step:

**Ask** (`POST /collections/{id}/ask`)

```
POST /api/v1/collections/{id}/ask                      (auto, FastAPI)
└─ rag.ask                       question length, collection, mode
   ├─ rag.search                 mode, top_k, candidates
   │  ├─ embeddings qwen3-embedding:8b      tokens
   │  ├─ db.vector_search        matches
   │  ├─ db.keyword_search       matches
   │  └─ rag.rerank              reranker, depth, top score, failed?
   ├─ rag.context                sources kept, dropped below bar, tokens
   ├─ chat gpt-oss:20b           input/output tokens, time to first token
   └─ rag.grounding              cited, invalid, revised?, withheld?
      └─ chat gpt-oss:20b        (only if the citation retry runs)
```

**Agent** (`/agent/ask`): `rag.agent` (run ID, tool-call limit) with, per turn, a `chat` span and an `execute_tool search_knowledge_base` span that contains the same `rag.search` tree as above; then `rag.grounding`.

**Search** (`/search`): the `rag.search` tree alone.

**Upload / ingestion job** (worker):

```
ingestion.job                    job ID, document ID, type, attempt
├─ ingestion.parse               parser (docling | pypdf), pages, sections, fallback used?
├─ ingestion.chunk               chunks, tokens
├─ embeddings qwen3-embedding:8b   (one span per batch)
└─ ingestion.store               chunks saved
```

The upload request and the worker job are separate traces (the job runs later). The job span stores the document ID, so you can find one from the other.

**Attributes follow the OpenTelemetry GenAI conventions** where they exist:

| Attribute | Example |
|---|---|
| `gen_ai.operation.name` | `chat`, `embeddings`, `execute_tool`, `invoke_agent` |
| `gen_ai.provider.name` | `ollama` |
| `gen_ai.request.model` / `gen_ai.response.model` | `gpt-oss:20b` |
| `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` | `1203` / `152` |
| `rag.*` (ours) | `rag.reranker = cross_encoder:Qwen/Qwen3-Reranker-0.6B`, `rag.sources.kept = 3`, `rag.answer.withheld = false` |

The GenAI conventions are still marked "development" by OpenTelemetry. Names may change between versions; we pin the package versions and follow upgrades.

### 4.2 Metrics

| Metric | Type | Labels | Answers |
|---|---|---|---|
| `http.server.request.duration` (auto) | histogram | route, method, status | Latency and error rate per endpoint |
| `gen_ai.client.operation.duration` | histogram | operation, model | How slow is each model call? |
| `gen_ai.client.token.usage` | histogram | type (input/output), model | Tokens used per call; tokens per hour |
| `rag.step.duration` | histogram | step (embed, vector_search, keyword_search, rerank, generate, repair) | Which step is slow? |
| `rag.answers` | counter | mode (ask/agent), outcome (cited, declined, no_sources, revised, withheld) | How often do the guards fire? |
| `rag.rerank.failures` | counter | reranker | Is the reranker breaking? |
| `rag.agent.tool_calls` | histogram | | How many searches does the agent need? |
| `ingestion.jobs` | counter | parser, outcome (succeeded, failed, retrying) | Are uploads failing? |
| `ingestion.job.duration` | histogram | parser, step (parse, chunk, embed, store) | How long does Docling take? |
| `ingestion.parse.fallbacks` | counter | | How often does Docling fall back to pypdf? |

Labels are kept low-cardinality: no question text, no IDs in metric labels (IDs belong in traces).

### 4.3 Logs

- Every log line gets `trace_id` and `span_id` (a structlog processor), so logs and traces link both ways.
- With Grafana enabled, logs are also exported to Loki over OTLP. They still go to the console as today. OTel log export in Python is still marked beta; if it causes trouble, we keep console logs only and still have the trace IDs.

### 4.4 Content capture and privacy

By default **no prompt, question, passage or answer text** is recorded: only sizes, counts, scores, models, tokens and timings.

`OBSERVABILITY_CAPTURE_CONTENT=true` adds the question, the prompt messages, the sources and the answer to the spans. This is what makes Langfuse really useful for debugging answers ("why did it cite [2]?").

- Captured content is sent **to Langfuse only**. The Grafana exporter removes those attributes before sending, so Tempo never stores document text.
- Long texts are truncated (e.g. 8 KB per field) to keep traces small.
- Recommended: on for local development, off anywhere with real user data until access to Langfuse is controlled.

---

## 5. What Grafana and Langfuse each get

Both receive the same **traces** from the app; only Grafana receives **metrics** and **logs** (Langfuse's OTel endpoint accepts traces only).

| | Grafana LGTM | Langfuse |
|---|---|---|
| **Receives** | Traces, metrics, logs | Traces only |
| **Stores in** | Tempo (traces), Prometheus (metrics), Loki (logs) | ClickHouse (traces/observations), Postgres (projects, users) |
| **Trace view** | Generic waterfall of all spans, including HTTP and SQL | LLM-focused: "generations" with model, tokens, prompt and answer; retrievers, tools, agent steps |
| **Content** (prompts, answers) | Never (stripped) | When `OBSERVABILITY_CAPTURE_CONTENT=true` |
| **Dashboards** | Our provisioned dashboard: latency per step, error rates, tokens, guard outcomes, ingestion | Built-in: traces over time, latency, token usage, cost per model |
| **Logs** | Yes, linked to traces | No |
| **Alerts** | Yes (Grafana alerting) | No |
| **Best for** | "Is the system healthy? Which step is slow? What's failing?" | "Why did *this* answer come out like this?" Prompt, sources, tokens |
| **Extra later** | | Scores: attach our evaluation results to traces; compare prompt versions |

How Langfuse reads our spans: it maps OTel attributes to its own model. Each span becomes an **observation** whose type comes from `langfuse.observation.type` (or is inferred from `gen_ai.operation.name`). We set:

| Our span | Langfuse type |
|---|---|
| `chat gpt-oss:20b` | `generation` (shows model, tokens, prompt and answer) |
| `embeddings qwen3-embedding:8b` | `embedding` |
| `rag.search`, `rag.rerank` | `retriever` |
| `execute_tool …` | `tool` |
| `rag.agent` | `agent` |
| `rag.grounding` | `guardrail` |
| everything else | `span` |

Each trace also gets `langfuse.trace.name` (`ask`, `agent`, `search`, `ingestion`), tags (collection name, reranker, parser), `langfuse.user.id` (the API key's name) and the app version (`langfuse.release`).

**Costs:** Langfuse computes cost from model prices. Our models run locally, so cost is 0 unless you define a custom price per model in Langfuse. Token counts are always shown.

---

## 6. Configuration

All settings live in `.env` (read by `app/core/config.py`, like every other setting). We use our own names, not the standard `OTEL_*` variables, so there is one place to configure and Grafana and Langfuse can be set separately.

| Setting | Default | Meaning |
|---|---|---|
| `OBSERVABILITY_ENABLED` | `false` | Master switch. `false`: nothing is recorded, no overhead. |
| `OTLP_ENDPOINT` | *(empty)* | Grafana's OTLP/HTTP address, e.g. `http://localhost:4318`. Empty: don't send to Grafana. |
| `LANGFUSE_BASE_URL` | *(empty)* | Langfuse address, e.g. `http://localhost:3001`. Empty: don't send to Langfuse. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | *(empty)* | Project API keys. The local Docker setup creates `pk-lf-local-dev` / `sk-lf-local-dev`. |
| `OBSERVABILITY_CAPTURE_CONTENT` | `false` | Send prompts, sources and answers to Langfuse (section 4.4). |
| `OBSERVABILITY_SAMPLE_RATIO` | `1.0` | Share of requests traced (1.0 = all). Lower it under heavy load. |
| `OBSERVABILITY_ENVIRONMENT` | `dev` | Tag on all data (`deployment.environment.name`), to tell dev from prod. |

The service name is set per process automatically: `agentic-rag-api`, `agentic-rag-worker`, `agentic-rag-evals`.

**Examples (`.env`):**

```bash
# Grafana only
OBSERVABILITY_ENABLED=true
OTLP_ENDPOINT=http://localhost:4318

# Langfuse only
OBSERVABILITY_ENABLED=true
LANGFUSE_BASE_URL=http://localhost:3001
LANGFUSE_PUBLIC_KEY=pk-lf-local-dev
LANGFUSE_SECRET_KEY=sk-lf-local-dev
OBSERVABILITY_CAPTURE_CONTENT=true

# Both: all five lines above
```

**App running in Docker** (`make up`): containers reach the host through `host.docker.internal`, so the main `docker-compose.yml` gets `OTLP_ENDPOINT=http://host.docker.internal:4318` and `LANGFUSE_BASE_URL=http://host.docker.internal:3001`, using the same `DOCKER_`-prefixed override pattern as the model URLs.

---

## 7. Docker setup

### Files

| File | What it runs |
|---|---|
| `observability/grafana/docker-compose.yml` | `grafana/otel-lgtm:0.35.0`: one container (Collector, Tempo, Prometheus, Loki, Pyroscope, Grafana) |
| `observability/grafana/dashboards/agentic-rag.json` | The "Agentic RAG" dashboard, loaded automatically (`dashboards.yaml` tells Grafana where it is) |
| `observability/langfuse/docker-compose.yml` | Langfuse v4: web, worker, Postgres 17, ClickHouse 25.12, Redis 7, MinIO |

**Why no Dockerfile:** both stacks use the official, published images unchanged; configuration happens through environment variables and mounted files. A custom Dockerfile would only be needed to bake our dashboard into an image, and mounting it is simpler. The app's own `Dockerfile` needs no change, because the OTel packages become base dependencies.

They are separate Compose projects from the app's `docker-compose.yml`, so you can start, stop or reset them independently, and the app also runs without Docker (`make run`).

### Ports

| Port | Service |
|---|---|
| 3000 | Grafana UI (login `admin` / `admin`; anonymous admin access is on for local use) |
| 4318 | OTLP/HTTP into Grafana's collector (the app sends here) |
| 4317 | OTLP/gRPC into Grafana's collector (unused by the app) |
| 127.0.0.1:9090 | Prometheus UI/API |
| 3001 | Langfuse UI and its OTel endpoint `/api/public/otel` |
| 9190 | Langfuse's MinIO (the browser loads media from it) |

Compared with the official Langfuse compose, we moved the UI to 3001 (Grafana has 3000) and MinIO to 9190 (Prometheus has 9090). Postgres, ClickHouse, Redis and the worker publish **no** ports, so they can't clash with your local Postgres on 5432.

### Commands

```bash
# Grafana (about 1–2 GB RAM)
docker compose -f observability/grafana/docker-compose.yml up -d
open http://localhost:3000

# Langfuse (about 4 GB+ RAM; first start takes a minute or two for migrations)
docker compose -f observability/langfuse/docker-compose.yml up -d
open http://localhost:3001        # admin@example.com / langfuse-dev-password

# Stop (data is kept in Docker volumes)
docker compose -f observability/grafana/docker-compose.yml down
docker compose -f observability/langfuse/docker-compose.yml down

# Reset (delete all stored telemetry)
docker compose -f observability/<grafana|langfuse>/docker-compose.yml down -v
```

Make targets: `make obs-grafana`, `make obs-langfuse`, `make obs-up` (both) and `make obs-down`.

### Secrets

Both files contain **local development defaults** (the same approach as the app's `docker-compose.yml`). Langfuse's passwords, keys and `ENCRYPTION_KEY` can be overridden through environment variables or a `.env` file next to its compose file. For anything other than your own machine, change all of them (`openssl rand -hex 32` for the encryption key), and note that `otel-lgtm` is meant for development and demos, not production.

---

## 8. How to use them

### Grafana

1. **Find a trace:** Explore → data source **Tempo** → Search → service `agentic-rag-api`, span name `rag.ask`; or filter "duration > 5s" to find slow questions. A trace opens as a waterfall: each step, its duration and attributes.
2. **From a log line:** Explore → **Loki** → `{service_name="agentic-rag-api"}` → click the trace ID on a line.
3. **Dashboard "Agentic RAG"** (provisioned with the implementation):
   - Requests per minute and error rate (Ask, Agent, Search)
   - p50/p95 latency per RAG step
   - Tokens per hour by model
   - Answer outcomes: cited / declined / no sources / revised / withheld
   - Reranker failures; agent tool calls per question
   - Ingestion: jobs by outcome and parser, time per step, Docling fallbacks
4. **Metrics ad hoc:** Explore → **Prometheus**, e.g. the 95th percentile of rerank time.
5. **Alerts (optional):** e.g. "withheld answers > 10% for 15 minutes".

### Langfuse

1. **Tracing → Traces:** one row per question, upload or search, with name, latency, tokens and tags. Filter by name (`ask`, `agent`), tag (collection) or user.
2. **Open a trace:** the tree of observations. Click the `generation` to see the prompt messages, the answer, model, tokens and latency (prompts and answers only with content capture). The `retriever` observations show which passages were found and their reranker scores.
3. **Dashboards:** latency, token usage and trace counts over time, per model.
4. **Later (not in this plan):** send evaluation results as Langfuse **scores**, then compare runs or prompt versions (`answer-v3` vs `answer-v4`).

### Both

Enable both destinations: Grafana for "is something wrong and where", Langfuse for "what exactly did the model see and say". A trace has the same trace ID in both.

---

## 9. Code design

### Architecture rules

`app/application` must not depend on frameworks (enforced by import-linter in CI). So the use cases don't import OpenTelemetry directly; they use a small **port**:

```python
# app/application/ports/telemetry.py
class Telemetry(Protocol):
    def span(self, name: str, kind: SpanKind = "span", **attributes: Any) -> ContextManager[Span]: ...
    def count(self, metric: str, value: int = 1, **labels: str) -> None: ...
    def record(self, metric: str, value: float, **labels: str) -> None: ...
```

- **`NoopTelemetry`** (the default) does nothing, so tests and disabled setups pay nothing.
- **`OpenTelemetryAdapter`** (`app/infrastructure/observability/`) implements it with the OTel SDK, sets the GenAI and Langfuse attributes, and handles content capture.
- The container builds one or the other from settings and passes it to the use cases, like every other dependency.

### Files

| File | Change |
|---|---|
| `app/core/config.py`, `.env.example` | The settings in section 6 |
| `app/application/ports/telemetry.py` | New: the port and `NoopTelemetry` |
| `app/infrastructure/observability/setup.py` | New: build tracer/meter/logger providers and exporters (Grafana, Langfuse with Basic auth), sampler, content-stripping exporter for Grafana, flush on shutdown |
| `app/infrastructure/observability/adapter.py` | New: `OpenTelemetryAdapter` |
| `app/main.py` | Set up at startup; FastAPI, SQLAlchemy and httpx instrumentation; flush at shutdown |
| `app/worker.py` | Set up for the worker; flush on stop |
| `app/core/logging.py` | Add `trace_id` / `span_id` to log lines; OTLP log handler when Grafana is on |
| `app/application/use_cases/retrieval`, `answering`, `agent`, `ingestion` | Spans and metrics around each step (section 4) |
| `app/infrastructure/llm/openai_chat.py`, `openai_embedder.py` | GenAI spans: model, tokens, time to first token; content when enabled |
| `app/infrastructure/reranking/*` | Rerank span attributes |
| `app/bootstrap/container.py` | Build the telemetry and inject it |
| `docker-compose.yml` | Pass the observability settings to the app containers |
| `Makefile` | `obs-grafana`, `obs-langfuse`, `obs-down` |
| `observability/grafana/dashboards/agentic-rag.json` | The dashboard |
| `README.md` | Short "Observability" section |

### Behaviour guarantees

- **Off by default.** Disabled: no exporters and no background threads, so no overhead.
- **Never breaks a request.** Export happens in a background batch processor. If Grafana or Langfuse is down, data is dropped with a warning; requests are unaffected.
- **Streaming stays correct.** The `chat` span ends when the stream ends, and records time to first token.
- **Flush on shutdown** for the API and the worker, so the last traces aren't lost.

---

## 10. Implementation steps

Each step ends with tests passing and something visible.

| Step | Deliverable | Visible result |
|---|---|---|
| 1. Foundation | Settings, port, no-op, OTel setup with both exporters, auto-instrumentation (FastAPI, SQLAlchemy, httpx), trace IDs in logs, Make targets | Every HTTP request, SQL query and Ollama call appears as a trace in Grafana and Langfuse |
| 2. Ask, Search, Agent | Spans and metrics for search, rerank, context, generate, grounding, agent turns and tools; GenAI attributes; Langfuse observation types | The span trees of section 4.1; generations with tokens in Langfuse |
| 3. Ingestion | Worker spans and metrics: parse (with parser and fallback), chunk, embed, store | Upload jobs traced; Docling time per document |
| 4. Content capture | Question, prompts, sources and answers on spans, Langfuse only, truncated | Full prompts and answers in Langfuse |
| 5. Dashboard and docs | Provisioned Grafana dashboard; README section; app containers configured | Open Grafana → "Agentic RAG" dashboard with live data |

Steps 1–3 are the core. Step 4 is small but needs the privacy decision. Step 5 is polish.

---

## 11. Testing

- **Unit:** OTel's in-memory span exporter and metric reader check that:
  - an Ask produces the expected span tree and attributes;
  - tokens and outcomes are recorded;
  - content is absent by default;
  - content is stripped from the Grafana stream even when capture is on;
  - the no-op path records nothing.
- **Failure:** with an unreachable endpoint, requests still succeed.
- **Manual end-to-end:** start both stacks, ask the "quick buck" question and upload a PDF, then check the trace in Grafana and in Langfuse, and the dashboard panels.
- CI needs no Docker containers for these tests.

---

## 12. Risks and decisions for review

| # | Topic | Proposal |
|---|---|---|
| 1 | **Content capture default** | Off by default; turned on in your local `.env`. Never sent to Grafana. |
| 2 | **Where the OTel packages go** | Base dependencies (small, needed in Docker), not an optional extra. |
| 3 | **One destination or both** | Code supports both; you choose in `.env`. Recommended start: Grafana always, Langfuse when debugging answers. |
| 4 | **GenAI conventions are still evolving** | Pin versions; attribute names may change on upgrades. |
| 5 | **Langfuse resource use** | About 4 GB+ RAM (ClickHouse). Start it only when needed. |
| 6 | **Langfuse version** | The official compose uses major tag `4`; we follow it. Pin an exact version if upgrades cause surprises. |
| 7 | **Evaluation results as Langfuse scores** | Out of scope here; a natural follow-up. |
| 8 | **Not production-hardened** | `otel-lgtm` is for development/demos. For production, use a managed backend (Grafana Cloud, Langfuse Cloud) or the separate Grafana components. App code is unchanged; only endpoints and keys differ. |

---

## 13. References

- OpenTelemetry: <https://opentelemetry.io/docs/what-is-opentelemetry/>
- OpenTelemetry Python: <https://opentelemetry.io/docs/languages/python/>
- GenAI semantic conventions: <https://opentelemetry.io/docs/specs/semconv/gen-ai/>
- Grafana `otel-lgtm` image: <https://github.com/grafana/docker-otel-lgtm>
- Langfuse self-hosting (Docker Compose): <https://langfuse.com/self-hosting/docker-compose>
- Langfuse OpenTelemetry endpoint and attribute mapping: <https://langfuse.com/integrations/native/opentelemetry>

---

## 14. Implementation notes: differences from the plan

| Topic | Plan | What was built, and why |
|---|---|---|
| Outgoing HTTP spans | `opentelemetry-instrumentation-httpx` | Dropped. The OpenAI SDK used here is built on `httpx2`, which that package doesn't instrument. Model calls are covered by our own `chat …` / `embeddings …` spans, which carry more detail (model, tokens, time to first token). |
| SQL spans | Automatic | Kept, with `skip_dep_check`: the instrumentation declares support up to SQLAlchemy 2.0 and we run 2.1. `tests/integration/test_observability.py` checks it works against a real Postgres. |
| Idle polling noise | Not covered | A sampler (`SkipOrphanClientCalls`) drops outgoing calls (SQL, HTTP) that aren't part of a request or job. Otherwise the worker's queue polling every 2 s and health checks would each create a one-span trace. |
| Streaming answers | Not covered | Ask and Agent stream their answers from async generators that resume in another task. Their root spans are started explicitly and made current only around awaits, never across a `yield` (see `ports/telemetry.py`); a test resumes a stream in a second task. |
| Token totals on root spans | `gen_ai.usage.*` | Root spans use `rag.usage.*`. Langfuse sums `gen_ai.usage.*` over a trace, so putting them on the root as well counted every token twice. |
| First counter event | Not covered | Grafana's Prometheus runs with `created-timestamp-zero-ingestion`. Without it, `increase()` and `rate()` ignore the first event of each new series, and a dev setup with few requests showed zeros. |
| Log export | structlog lines as-is | A log handler subclass drops private record fields: structlog attaches its logger object (`_logger`), which isn't a valid attribute value and produced a warning per line. |
| Export connections | Exporter defaults (kept alive) | Every export batch opens a fresh connection (`setup.fresh_connections`). With Rancher Desktop's port forwarding, idle kept-alive connections died silently and left long-running processes stuck, no longer exporting, while new processes worked. A local connection per batch costs well under a millisecond. |
| Langfuse API | v3 trace API | Langfuse v4 ("events only") replaced `GET /api/public/traces`; scripts that read traces use `GET /api/public/v2/observations?traceId=…`. The UI is unaffected. |

**Verified end to end** on a laptop (M1 Max, Docker with 6 GB): Ask, Agent, Search and an upload job, with traces in Tempo and Langfuse, metrics in Prometheus, logs in Loki (with trace IDs), and every dashboard query returning data. Both stacks together used about 3 GB of memory.
